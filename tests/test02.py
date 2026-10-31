# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import onnxruntime as ort

# ============ 配置 ============
DET_ONNX = r"D:\workspace\py_stu\models\det.onnx"  # 你的检测模型
TEST_IMG = r"D:\workspace\py_stu\tests\test.jpg"
USE_GPU = True
INPUT_SIZE = (960, 736)  # (w, h)，PP-OCR 常用 960x736；也可用 (640, 640)
THRESH = 0.3  # 二值化阈值
BOX_THRESH = 0.6  # 文本框置信度阈值
MIN_AREA = 100  # 过滤太小的框
UNCLIP_RATIO = 1.6  # 文本框扩张倍率


# ============ 1. 会话创建（GPU 自动回退 CPU） ============
def create_session(onnx_path, use_gpu=False):
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"模型不存在: {onnx_path}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4

    if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        print("[info] 使用 GPU 推理")
    else:
        providers = ["CPUExecutionProvider"]
        print("[info] 使用 CPU 推理, 可用 EP:", ort.get_available_providers())

    sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
    inp = sess.get_inputs()[0]
    print(f"[info] 模型输入 name={inp.name}, shape={inp.shape}, type={inp.type}")
    for i, o in enumerate(sess.get_outputs()):
        print(f"[info] 模型输出[{i}] name={o.name}, shape={o.shape}")
    return sess


# ============ 2. 预处理 ============
def preprocess(img, target_size):
    """等比缩放 + padding 到 target_size，返回 img_pad 和缩放比"""
    h, w = img.shape[:2]
    tw, th = target_size

    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.ones((th, tw, 3), dtype=np.uint8) * 0
    padded[:nh, :nw] = resized

    pad_img = padded.astype("float32")
    # PP-OCR 均值/std；如果是自己训练的 DBNet 可能只需 /255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    pad_img = (pad_img / 255.0 - mean) / std
    pad_img = pad_img.transpose(2, 0, 1)[None].astype("float32")  # (1,3,H,W)
    return pad_img, scale


# ============ 3. DB 后处理 ============
def unclip(box, unclip_ratio):
    """多边形外扩"""
    poly = box.reshape(-1, 2)
    area = cv2.contourArea(poly)
    length = cv2.arcLength(poly, True)
    if length == 0:
        return box
    distance = area * unclip_ratio / length
    return np.array(cv2.approxPolyDP(
        cv2.approxPolyDP(poly, 2, True) +
        (np.outer(np.concatenate([np.ones(len(poly)), np.ones(len(poly))]), [distance])
         if False else poly), 2, True))  # 占位，下面用标准写法


def get_mini_boxes(contour):
    """从轮廓取最小外接矩形，返回四点 + 短边长度"""
    rect = cv2.minAreaRect(contour)
    pts = cv2.boxPoints(rect)
    pts = pts.astype(np.int32)
    idx = np.argsort(pts[:, 0])
    left, right = pts[idx[0]], pts[idx[1]]
    # 按 x 排序后取最左最右两个点，再按 y 区分上下
    order = np.argsort(left[:, 0])  # 简化：直接用 boxPoints 结果排序
    pts_sorted = sorted(pts, key=lambda p: p[0])
    p1, p2 = pts_sorted[0], pts_sorted[-1]
    short_edge = min(rect[1])
    return np.array(sorted(pts, key=lambda x: x[0])), short_edge


def box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
    """
    掩码方式计算四边形框内bitmap均值
    :param bitmap: H,W 灰度图, uint8 0~255
    :param box: (4,2) float32 四点四边形
    :return: 框内mask区域平均灰度
    """
    h, w = bitmap.shape[:2]
    xmin = max(0, int(box[:, 0].min()))
    xmax = min(w - 1, int(box[:, 0].max()))
    ymin = max(0, int(box[:, 1].min()))
    ymax = min(h - 1, int(box[:, 1].max()))

    if xmax <= xmin or ymax <= ymin:
        return 0.0

    roi_h = ymax - ymin + 1
    roi_w = xmax - xmin + 1
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

    # 坐标平移到roi局部坐标系
    shifted = box.copy()
    shifted[:, 0] -= xmin
    shifted[:, 1] -= ymin

    cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)

    # ✅ 修正切片 ymin:ymax+1 , xmin:xmax+1
    roi = bitmap[ymin:ymax + 1, xmin:xmax + 1]
    mean_val = cv2.mean(roi, mask)[0]
    return float(mean_val)


def polygons_from_bitmap(bitmap, orig_h, orig_w, scale):
    """概率图 -> 多边形框列表 (N, 4, 2)，坐标还原到原图"""
    bitmap = (bitmap * 255).astype(np.uint8)
    _, binary = cv2.threshold(bitmap, THRESH * 255, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 4:
            continue

        # 外接矩形快速打分
        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect).astype(np.int32)
        score = box_score_fast((bitmap / 255.0).astype(np.float32), box.astype(np.float32))
        if score < BOX_THRESH:
            continue

        # 外扩
        poly = approx.reshape(-1, 2).astype(np.float32)
        area = cv2.contourArea(poly)
        length = cv2.arcLength(poly, True)
        if length > 0:
            distance = area * UNCLIP_RATIO / length
            from numpy.linalg import norm
            # 简单外扩：沿法线方向平移（用 cv2.approxPolyDP 的简化版）
            center = poly.mean(axis=0)
            vecs = poly - center
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
            poly = poly + vecs / norms * distance

        # 还原到原图坐标
        poly /= scale
        poly = np.clip(poly, 0, 1e9)
        # 转成左上/右上/右下/左下顺序（按角度排序）
        cx, cy = poly.mean(axis=0)
        angles = np.arctan2(poly[:, 1] - cy, poly[:, 0] - cx)
        poly = poly[np.argsort(angles)]
        # 取四个角
        if len(poly) > 4:
            # 重新取最小外接矩形保证四边形
            rect = cv2.minAreaRect(poly)
            poly = cv2.boxPoints(rect)
        boxes.append((np.float32(poly), score))
    return boxes


def postprocess(out, img_shape, scale):
    """
    out: 模型输出，可能是 (1,1,H,W) 概率图，或 (1,2,H,W) [prob, thresh]
    """
    h, w = img_shape[:2]
    if isinstance(out, list):
        prob_map = out[0]
    else:
        prob_map = out
    if prob_map.ndim == 4:
        prob_map = prob_map[0]
    if prob_map.shape[0] > 1:  # 有 thresh 分支，取第一通道
        prob_map = prob_map[0]
    prob_map = prob_map.transpose(1, 2, 0)
    # 如果模型没带 sigmoid，这里补一个
    if prob_map.max() <= 1.0 and prob_map.min() >= 0:
        pass
    else:
        prob_map = 1 / (1 + np.exp(-prob_map))

    return polygons_from_bitmap(prob_map, h, w, scale)


    




# ============ 4. 可视化 ============
def draw_boxes(img, boxes):
    out = img.copy()
    for poly, score in boxes:
        pts = poly.astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        cv2.putText(out, f"{score:.2f}", tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out


def show_or_save(name, img):
    """headless 环境自动降级为保存图片"""
    try:
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        path = f"debug_{name}.jpg"
        cv2.imwrite(path, img)
        print(f"[warn] imshow 不可用，已保存到 {path}")


# ============ 5. 主流程 ============
def detect_ocr(img_path, use_gpu=False):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"读图失败: {img_path}")
    print(f"[info] 原图尺寸: {img.shape[1]}x{img.shape[0]}")

    sess = create_session(DET_ONNX, use_gpu)
    inp_name = sess.get_inputs()[0].name

    pad_img, scale = preprocess(img, INPUT_SIZE)
    print(f"[info] 输入shape: {pad_img.shape}, scale: {scale:.4f}")

    outs = sess.run(None, {inp_name: pad_img})

    boxes = postprocess(outs, img.shape, scale)
    print(f"[info] 检测到 {len(boxes)} 个文本框")

    vis = draw_boxes(img, boxes)
    show_or_save("result", vis)
    return boxes


if __name__ == "__main__":
    boxes = detect_ocr(TEST_IMG, use_gpu=USE_GPU)
    for i, (poly, score) in enumerate(boxes):
        print(f"box[{i}] score={score:.3f} pts={np.int32(poly).tolist()}")
