"""
test4insightface.py -- 调用 insightface 第三方库进行 SCRFD 人脸检测推理

调用链路:
    insightface.model_zoo.get_model(onnx_path)
        -> ModelRouter -> RetinaFace 类 (输出数>=5)
        -> .prepare()  设置设备/输入尺寸/阈值
        -> .detect()   letterbox resize + blobFromImage + session.run + 解码 + NMS
        -> 返回 bboxes(N,5)[x1,y1,x2,y2,score], kpss(N,5,2)[5个关键点]

依赖:
    pip install insightface onnxruntime opencv-python numpy
"""

import os
import sys
import cv2
import numpy as np
from insightface.model_zoo import get_model

# ============================================================
# 硬编码参数
# ============================================================
MODEL_PATH   = "det_10g.onnx"
IMAGES_DIR   = "."
OUTPUT_DIR   = "output_insightface"
INPUT_SIZE   = (640, 640)
CONF_THRESH  = 0.5
NMS_THRESH   = 0.4
CTX_ID       = -1          # -1=CPU, 0=GPU:0

# landmark 颜色 (BGR)
LMK_COLORS = [
    (0,   0,   255),  # 左眼:   红
    (255, 0,   0),    # 右眼:   蓝
    (0,   255, 0),    # 鼻子:   绿
    (0,   255, 255),  # 左嘴角: 黄
    (255, 0,   255),  # 右嘴角: 品红
]


def draw_detections(bgr, bboxes, kpss):
    for i in range(bboxes.shape[0]):
        x1, y1, x2, y2, score = bboxes[i]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(bgr, f"{score:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if kpss is not None:
            kps = kpss[i]
            for j in range(5):
                cx, cy = int(kps[j][0]), int(kps[j][1])
                cv2.circle(bgr, (cx, cy), 3, LMK_COLORS[j], -1)

    return bgr


def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"FATAL: model not found: {MODEL_PATH}")
        return 1
    if not os.path.isdir(IMAGES_DIR):
        print(f"FATAL: image dir not found: {IMAGES_DIR}")
        return 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载 SCRFD 模型
    detector = get_model(MODEL_PATH)
    detector.prepare(
        ctx_id=CTX_ID,
        input_size=INPUT_SIZE,
        det_thresh=CONF_THRESH,
        nms_thresh=NMS_THRESH,
    )

    # 扫描图片
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if os.path.splitext(f)[1].lower() in exts
    )
    if not images:
        print(f"FATAL: no images in {IMAGES_DIR}")
        return 1

    print(f"Model: {MODEL_PATH}")
    print(f"Images: {len(images)}, Output: {OUTPUT_DIR}")
    print(f"InputSize: {INPUT_SIZE}, Conf: {CONF_THRESH}, NMS: {NMS_THRESH}")
    print()

    total_faces = 0
    for idx, fname in enumerate(images):
        bgr = cv2.imread(os.path.join(IMAGES_DIR, fname), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[{idx+1}/{len(images)}] {fname}: FAILED to load")
            continue

        # 推理: 内部完成 letterbox + normalize + forward + decode + NMS
        bboxes, kpss = detector.detect(bgr, max_num=0, metric="default")

        n = bboxes.shape[0]
        total_faces += n

        draw_detections(bgr, bboxes, kpss)

        out_path = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(out_path, bgr)
        print(f"[{idx+1}/{len(images)}] {fname}: {n} face(s) -> {out_path}")

    print(f"\nDone. Total faces: {total_faces}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
