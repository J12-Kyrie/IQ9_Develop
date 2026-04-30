"""
test4onnx.py -- SCRFD ONNX 模型精度验证 (Windows)

用相同的图片 + SCRFD ONNX 模型推理, 绘制 bbox + landmarks, 与板上 QNN 结果对比.
预处理/解码逻辑与板上 C++ 代码完全一致.

用法:
    python test4onnx.py --model scrfd_10g_bnkps.onnx --images .  [--conf 0.5] [--nms 0.4]
    python test4onnx.py --model scrfd_2.5g_bnkps.onnx --images . [--conf 0.5] [--nms 0.4]

输出:
    <images>/output_onnx/*.jpg

依赖:
    pip install onnxruntime opencv-python numpy
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# ============================================================
# SCRFD 常量 (匹配 FaceTypes.hpp)
# ============================================================
INPUT_SIZE = 640
STRIDES = [8, 16, 32]
FMC = 2  # anchors per grid cell

# Landmark 颜色 (BGR, 匹配 test_scrfd_draw.cpp)
LMK_COLORS = [
    (0, 0, 255),    # 左眼: 红
    (255, 0, 0),    # 右眼: 蓝
    (0, 255, 0),    # 鼻子: 绿
    (0, 255, 255),  # 左嘴角: 黄
    (255, 0, 255),  # 右嘴角: 品红
]


# ============================================================
# 预处理: letterbox + normalize (匹配 FacePreprocess.cpp + scrfd_preprocess.cl)
# ============================================================
def preprocess(bgr: np.ndarray):
    """
    BGR 图 -> NCHW float32 (1,3,640,640)
    letterbox 放在左上角, padding 在右/下 (与 scrfd_preprocess.cl 一致)
    归一化: (x - 127.5) / 128.0
    返回: (input_tensor, scale, new_w, new_h)
    """
    h, w = bgr.shape[:2]
    im_ratio = h / w

    if im_ratio > 1.0:
        new_h = INPUT_SIZE
        new_w = int(INPUT_SIZE / im_ratio)
    else:
        new_w = INPUT_SIZE
        new_h = int(INPUT_SIZE * im_ratio)

    new_w = min(new_w, INPUT_SIZE)
    new_h = min(new_h, INPUT_SIZE)
    scale = w / new_w

    # resize (BGR -> resized BGR)
    resized = cv2.resize(bgr, (new_w, new_h))

    # BGR -> RGB
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 创建 640x640 画布, 填充 pad 值 = (0 - 127.5) / 128.0 不需要,
    # 先用 0 填充, 归一化后 pad 区域 = (0 - 127.5)/128 = -0.99609375
    # 但更简单: 先填充原始像素值 0, 再统一归一化
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    canvas[:new_h, :new_w, :] = resized_rgb

    # float32 + normalize: (x - 127.5) / 128.0
    blob = canvas.astype(np.float32)
    blob = (blob - 127.5) / 128.0

    # HWC -> NCHW
    blob = blob.transpose(2, 0, 1)  # CHW
    blob = blob[np.newaxis, ...]    # NCHW (1,3,640,640)

    return blob, scale, new_w, new_h


# ============================================================
# 自动映射 ONNX 输出 tensor (按 shape 识别)
# ============================================================
def build_tensor_map(session):
    """
    自动识别 ONNX 输出 tensor 的类型和 stride:
      - last_dim=1 -> score
      - last_dim=4 -> bbox
      - last_dim=10 -> kps
      - anchor_count -> stride (12800=s8, 3200=s16, 800=s32)

    返回: dict[stride] = {"score_idx": i, "bbox_idx": j, "kps_idx": k}
    """
    outputs = session.get_outputs()
    print(f"\n  ONNX 模型有 {len(outputs)} 个输出:")

    # 预计算每个 stride 的 anchor 数
    anchor_counts = {}
    for s in STRIDES:
        feat_size = INPUT_SIZE // s
        anchor_counts[feat_size * feat_size * FMC] = s

    tensor_map = {s: {} for s in STRIDES}

    for i, out in enumerate(outputs):
        shape = out.shape
        name = out.name
        # 获取有效维度 (去掉 batch=1)
        dims = [d for d in shape if d != 1 or len(shape) <= 2]
        if len(dims) == 0:
            print(f"    [{i}] {name}: shape={shape} -- SKIPPED (scalar)")
            continue

        # 找 anchor count 和 feature dim
        n_anchors = None
        feat_dim = None
        for d in shape:
            if d in anchor_counts:
                n_anchors = d
            if d in (1, 4, 10):
                feat_dim = d

        if n_anchors is None or feat_dim is None:
            print(f"    [{i}] {name}: shape={shape} -- SKIPPED (unrecognized)")
            continue

        stride = anchor_counts[n_anchors]
        if feat_dim == 1:
            tensor_map[stride]["score_idx"] = i
            kind = "score"
        elif feat_dim == 4:
            tensor_map[stride]["bbox_idx"] = i
            kind = "bbox"
        elif feat_dim == 10:
            tensor_map[stride]["kps_idx"] = i
            kind = "kps"
        else:
            kind = "?"

        print(f"    [{i}] {name}: shape={shape} -> stride={stride}, type={kind}")

    # 验证完整性
    for s in STRIDES:
        for key in ("score_idx", "bbox_idx", "kps_idx"):
            if key not in tensor_map[s]:
                print(f"\n  ERROR: stride={s} 缺少 {key}, 无法解码!")
                sys.exit(1)

    return tensor_map


# ============================================================
# SCRFD 解码 (匹配 ScrfdDecode.cpp)
# ============================================================
def scrfd_decode(outputs, tensor_map, scale, orig_w, orig_h, conf_thresh):
    """
    解码 SCRFD 输出 tensor, 返回 detections list:
    [(x1, y1, x2, y2, score, landmarks[5][2]), ...]
    """
    dets = []

    for stride in STRIDES:
        feat_w = INPUT_SIZE // stride
        sidx = tensor_map[stride]["score_idx"]
        bidx = tensor_map[stride]["bbox_idx"]
        kidx = tensor_map[stride]["kps_idx"]

        scores_raw = outputs[sidx].flatten()
        bboxes_raw = outputs[bidx].reshape(-1, 4)
        kps_raw = outputs[kidx].reshape(-1, 10)

        n_anchors = len(scores_raw)

        for i in range(n_anchors):
            score = float(scores_raw[i])

            # 自适应 sigmoid (匹配 ScrfdDecode.cpp:82-85)
            if score > 1.0 or score < 0.0:
                if score < -10.0:
                    continue
                score = 1.0 / (1.0 + np.exp(-score))

            if score < conf_thresh:
                continue

            # 锚点位置 (匹配 ScrfdDecode.cpp:90-94)
            point_idx = i // FMC
            ax = point_idx % feat_w
            ay = point_idx // feat_w
            anchor_cx = (ax + 0.5) * stride
            anchor_cy = (ay + 0.5) * stride

            # distance bbox 解码 (匹配 ScrfdDecode.cpp:97-101)
            bbox = bboxes_raw[i]
            x1 = (anchor_cx - bbox[0] * stride) * scale
            y1 = (anchor_cy - bbox[1] * stride) * scale
            x2 = (anchor_cx + bbox[2] * stride) * scale
            y2 = (anchor_cy + bbox[3] * stride) * scale

            # clamp
            x1 = max(0.0, min(x1, orig_w))
            y1 = max(0.0, min(y1, orig_h))
            x2 = max(0.0, min(x2, orig_w))
            y2 = max(0.0, min(y2, orig_h))

            # landmark 解码 (匹配 ScrfdDecode.cpp:116-118)
            lm = kps_raw[i]
            landmarks = []
            for j in range(5):
                lmk_x = (lm[j * 2] * stride + anchor_cx) * scale
                lmk_y = (lm[j * 2 + 1] * stride + anchor_cy) * scale
                landmarks.append((lmk_x, lmk_y))

            dets.append((x1, y1, x2, y2, score, landmarks))

    return dets


# ============================================================
# Greedy NMS (匹配 ScrfdDecode.cpp nms())
# ============================================================
def nms(dets, nms_thresh):
    if len(dets) == 0:
        return []

    # 按 score 降序排序
    dets = sorted(dets, key=lambda d: d[4], reverse=True)

    keep = []
    suppressed = [False] * len(dets)

    for i in range(len(dets)):
        if suppressed[i]:
            continue
        keep.append(dets[i])
        for j in range(i + 1, len(dets)):
            if suppressed[j]:
                continue
            # IoU
            ix1 = max(dets[i][0], dets[j][0])
            iy1 = max(dets[i][1], dets[j][1])
            ix2 = min(dets[i][2], dets[j][2])
            iy2 = min(dets[i][3], dets[j][3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_i = (dets[i][2] - dets[i][0]) * (dets[i][3] - dets[i][1])
            area_j = (dets[j][2] - dets[j][0]) * (dets[j][3] - dets[j][1])
            iou = inter / (area_i + area_j - inter + 1e-6)
            if iou > nms_thresh:
                suppressed[j] = True

    return keep


# ============================================================
# 绘图 (匹配 test_scrfd_draw.cpp)
# ============================================================
def draw_detections(bgr, dets):
    for (x1, y1, x2, y2, score, landmarks) in dets:
        # bbox: 绿色矩形, 线宽 2
        cv2.rectangle(bgr,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0, 255, 0), 2)

        # score: 白色文字
        label = f"{score:.2f}"
        cv2.putText(bgr, label,
                    (int(x1), max(int(y1) - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        # 5 landmarks: 彩色圆点, 半径 3
        for j in range(5):
            cx, cy = int(landmarks[j][0]), int(landmarks[j][1])
            cv2.circle(bgr, (cx, cy), 3, LMK_COLORS[j], -1)

    return bgr


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SCRFD ONNX 精度验证")
    parser.add_argument("--model", required=True, help="SCRFD ONNX 模型路径")
    parser.add_argument("--images", required=True, help="图片目录")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值 (default: 0.5)")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS 阈值 (default: 0.4)")
    args = parser.parse_args()

    # ---- 检查路径 ----
    if not os.path.isfile(args.model):
        print(f"FATAL: 模型文件不存在: {args.model}")
        return 1
    if not os.path.isdir(args.images):
        print(f"FATAL: 图片目录不存在: {args.images}")
        return 1

    # ---- 扫描图片 ----
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = sorted([f for f in os.listdir(args.images)
                     if os.path.splitext(f)[1].lower() in exts])
    if not images:
        print(f"FATAL: 目录 {args.images} 中无图片文件")
        return 1

    # ---- 创建输出目录 ----
    out_dir = os.path.join(args.images, "output_onnx")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 加载 ONNX 模型 ----
    print("=" * 50)
    print(" SCRFD ONNX Accuracy Test")
    print("=" * 50)
    print(f"  Model:    {args.model}")
    print(f"  Images:   {len(images)} file(s) in {args.images}")
    print(f"  Conf:     {args.conf}")
    print(f"  NMS:      {args.nms}")
    print(f"  Output:   {out_dir}")

    try:
        import onnxruntime as ort
    except ImportError:
        print("\nFATAL: onnxruntime 未安装, 请执行: pip install onnxruntime")
        return 1

    print("\n  加载模型...")
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    # 打印输入信息
    inp = sess.get_inputs()[0]
    print(f"  输入: {inp.name}, shape={inp.shape}, dtype={inp.type}")

    # 自动映射输出 tensor
    tensor_map = build_tensor_map(sess)
    print()

    # ---- 推理循环 ----
    input_name = inp.name
    total_faces = 0
    total_ms = 0.0
    failed = 0
    saved = 0

    for idx, fname in enumerate(images):
        fpath = os.path.join(args.images, fname)
        bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[{idx+1}/{len(images)}] {fname}: FAILED to load, skipping")
            failed += 1
            continue

        h, w = bgr.shape[:2]

        # 预处理
        blob, scale, new_w, new_h = preprocess(bgr)

        # 推理
        t0 = time.perf_counter()
        outputs = sess.run(None, {input_name: blob})
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        total_ms += ms

        # 解码
        dets = scrfd_decode(outputs, tensor_map, scale, w, h, args.conf)
        dets = nms(dets, args.nms)
        count = len(dets)
        total_faces += count

        # 绘图
        draw_detections(bgr, dets)

        # 保存
        out_path = os.path.join(out_dir, fname)
        if cv2.imwrite(out_path, bgr):
            saved += 1
        else:
            print(f"[{idx+1}/{len(images)}] {fname}: imwrite failed!")

        print(f"[{idx+1}/{len(images)}] {fname}: {count} face(s), "
              f"{ms:.1f}ms, scale={scale:.2f} ({w}x{h}->{new_w}x{new_h}) "
              f"-> {out_path}")

    # ---- 统计 ----
    processed = len(images) - failed
    print(f"\n{'=' * 50}")
    print(f" Summary")
    print(f"{'=' * 50}")
    print(f"  Total images:   {len(images)}")
    print(f"  Processed:      {processed}")
    print(f"  Failed:         {failed}")
    print(f"  Total faces:    {total_faces}")
    print(f"  Saved images:   {saved}")
    if processed > 0:
        print(f"  Avg faces/img:  {total_faces / processed:.1f}")
        print(f"  Avg time/img:   {total_ms / processed:.1f} ms")
        print(f"  Total time:     {total_ms:.0f} ms")
    print(f"  Output dir:     {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
