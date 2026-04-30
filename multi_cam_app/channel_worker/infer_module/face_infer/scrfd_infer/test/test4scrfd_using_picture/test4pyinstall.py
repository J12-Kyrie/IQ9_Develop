"""
test_insightface_debug.py -- InsightFace 官方库推理 + 拦截中间结果导出

特点：
1. 完全使用 insightface 官方库加载模型和前处理/后处理。
2. 参数采用 Python 顶部硬编码。
3. 利用 Hook 机制动态拦截底层 ONNX 推理，无损导出中间 Tensor (Score/Bbox/Kps)。
"""

import os
import cv2
import numpy as np
import insightface
from insightface.model_zoo import get_model

# ============================================================
# 1. 硬编码参数配置 (在这里修改)
# ============================================================
MODEL_PATH = "scrfd_10g_bnkps.onnx"  # [请修改] SCRFD ONNX 模型路径
IMAGES_DIR = "."  # [请修改] 图片所在目录
CONF_THRESH = 0.5  # 置信度阈值
NMS_THRESH = 0.4  # NMS 阈值

# 中间结果调试开关
EXPORT_INTERMEDIATE = True  # 是否导出模型的原始推理结果(中间Tensor)
EXPORT_FORMAT = "npy"  # 导出格式: 'npy' 或 'txt'

# ============================================================
# SCRFD 常量及颜色配置
# ============================================================
INPUT_SIZE = 640
STRIDES = [8, 16, 32]
FMC = 2

LMK_COLORS = [
    (0, 0, 255),  # 左眼: 红
    (255, 0, 0),  # 右眼: 蓝
    (0, 255, 0),  # 鼻子: 绿
    (0, 255, 255),  # 左嘴角: 黄
    (255, 0, 255),  # 右嘴角: 品红
]


# ============================================================
# 分析 InsightFace 底层的 ONNX Session，映射输出结构
# ============================================================
def build_tensor_map(session):
    outputs = session.get_outputs()
    anchor_counts = {}
    for s in STRIDES:
        feat_size = INPUT_SIZE // s
        anchor_counts[feat_size * feat_size * FMC] = s

    tensor_map = {s: {} for s in STRIDES}
    for i, out in enumerate(outputs):
        shape = out.shape
        dims = [d for d in shape if d != 1 or len(shape) <= 2]
        if len(dims) == 0: continue

        n_anchors, feat_dim = None, None
        for d in shape:
            if d in anchor_counts: n_anchors = d
            if d in (1, 4, 10): feat_dim = d

        if n_anchors is None or feat_dim is None: continue

        stride = anchor_counts[n_anchors]
        if feat_dim == 1:
            tensor_map[stride]["score_idx"] = i
        elif feat_dim == 4:
            tensor_map[stride]["bbox_idx"] = i
        elif feat_dim == 10:
            tensor_map[stride]["kps_idx"] = i

    return tensor_map


# ============================================================
# 导出中间结果
# ============================================================
def export_intermediate_tensors(outputs, tensor_map, fname, out_dir):
    base_name = os.path.splitext(fname)[0]
    for stride in STRIDES:
        sidx = tensor_map[stride]["score_idx"]
        bidx = tensor_map[stride]["bbox_idx"]
        kidx = tensor_map[stride]["kps_idx"]

        data_dict = {
            "score": outputs[sidx],
            "bbox": outputs[bidx],
            "kps": outputs[kidx]
        }

        for kind, tensor in data_dict.items():
            out_name = f"{base_name}_stride{stride}_{kind}"
            out_path = os.path.join(out_dir, out_name)
            if EXPORT_FORMAT == "npy":
                np.save(out_path + ".npy", tensor)
            else:
                np.savetxt(out_path + ".txt", tensor.flatten(), fmt='%.6f')


# ============================================================
# 绘图逻辑
# ============================================================
def draw_detections(bgr, bboxes, kpss):
    if bboxes is None or bboxes.shape[0] == 0:
        return bgr

    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        score = bbox[4]

        if score < CONF_THRESH:
            continue

        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{score:.2f}"
        cv2.putText(bgr, label, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if kpss is not None:
            kps = kpss[i]
            for j in range(5):
                cx, cy = int(kps[j][0]), int(kps[j][1])
                cv2.circle(bgr, (cx, cy), 3, LMK_COLORS[j], -1)

    return bgr


# ============================================================
# main
# ============================================================
def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"FATAL: 模型文件不存在: {MODEL_PATH}")
        return 1
    if not os.path.isdir(IMAGES_DIR):
        print(f"FATAL: 图片目录不存在: {IMAGES_DIR}")
        return 1

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = sorted([f for f in os.listdir(IMAGES_DIR) if os.path.splitext(f)[1].lower() in exts])

    out_dir_img = os.path.join(IMAGES_DIR, "output_insightface_images")
    os.makedirs(out_dir_img, exist_ok=True)

    if EXPORT_INTERMEDIATE:
        out_dir_tensor = os.path.join(IMAGES_DIR, "output_insightface_tensors")
        os.makedirs(out_dir_tensor, exist_ok=True)

    print("\n调用 insightface 加载模型...")
    # 1. 使用 insightface 官方库加载
    model = get_model(MODEL_PATH)
    # ctx_id=-1 代表使用CPU。限制输入尺寸为了方便对其 anchor count
    model.prepare(ctx_id=-1, nms_thresh=NMS_THRESH, input_size=(INPUT_SIZE, INPUT_SIZE), det_thresh=CONF_THRESH)

    # 获取底层的 onnxruntime session，解析输出结构
    tensor_map = build_tensor_map(model.session)
    # 保存 insightface 原始的 session.run 方法
    original_run = model.session.run

    for idx, fname in enumerate(images):
        fpath = os.path.join(IMAGES_DIR, fname)
        bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if bgr is None: continue

        # 存放本次推理截获的中间结果
        captured_outputs = []

        # ==================== 核心：拦截 Hook ====================
        def hooked_run(output_names, input_feed, run_options=None):
            # 正常执行底层推理
            outputs = original_run(output_names, input_feed, run_options)
            # 偷偷把拿到的输出存进我们的列表里
            captured_outputs.append(outputs)
            return outputs

        # 替换 insightface 的底层方法
        model.session.run = hooked_run
        # =========================================================

        # 2. 调用 insightface 官方的高级 API 进行检测 (内置预处理、推理、后处理)
        bboxes, kpss = model.detect(bgr, max_num=0, metric='default')

        # 恢复正常的底层方法 (防止内存泄漏或影响后续)
        model.session.run = original_run

        # 3. 导出拦截到的中间结果
        if EXPORT_INTERMEDIATE and len(captured_outputs) > 0:
            # captured_outputs[0] 就是本次图片输入给网络后，吐出的最原始的 raw tensors
            export_intermediate_tensors(captured_outputs[0], tensor_map, fname, out_dir_tensor)

        # 4. 绘制并保存最终结果
        bgr_drawn = draw_detections(bgr, bboxes, kpss)
        out_path = os.path.join(out_dir_img, fname)
        cv2.imwrite(out_path, bgr_drawn)

        face_count = bboxes.shape[0] if bboxes is not None else 0
        print(f"[{idx + 1}/{len(images)}] 处理完毕: {fname} (检测到 {face_count} 个人脸)")

    print("\n全部处理完毕！")
    if EXPORT_INTERMEDIATE:
        print(f"-> 拦截到的网络原始中间 Tensor 结果已保存至: {out_dir_tensor}")
    print(f"-> 渲染画框结果已保存至: {out_dir_img}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main() or 0)