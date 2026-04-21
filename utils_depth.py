# utils_depth.py
import depth_pro
from PIL import Image
import numpy as np
import torch
import cv2

def load_depth_pro_model():
    """加载Depth Pro模型和预处理转换函数"""
    print("正在加载Depth Pro模型，请稍候...")
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    print("Depth Pro模型加载成功！")
    return model, transform

def generate_depth_map(image_path, model, transform):
    """
    输入：图片路径, 模型, 预处理函数
    输出：深度图 (numpy数组，单位：米), 焦距 f_px (像素)
    """
    image, _, f_px = depth_pro.load_rgb(image_path)
    image_tensor = transform(image)
    with torch.no_grad():
        prediction = model.infer(image_tensor, f_px=f_px)
    depth_map_m = prediction["depth"].cpu().numpy().squeeze()
    return depth_map_m, f_px

def get_lesion_mask(image, ort_session):
    """
    使用你训练好的U-Net模型，生成伤口的二值掩码。
    输入：原始BGR图片, ONNX Runtime会话
    输出：二值掩码 (伤口区域为255，背景为0)
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    TARGET_SIZE = 512
    scale = TARGET_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_rgb, (new_w, new_h))
    pad_h = (TARGET_SIZE - new_h) // 2
    pad_w = (TARGET_SIZE - new_w) // 2
    img_padded = cv2.copyMakeBorder(img_resized, pad_h, TARGET_SIZE-new_h-pad_h, pad_w, TARGET_SIZE-new_w-pad_w, cv2.BORDER_CONSTANT, value=0)
    
    img_norm = img_padded.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_norm - mean) / std
    img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    
    outputs = ort_session.run(['output'], {'input': img_input})[0]
    pred_mask = (outputs[0, 0] > 0.5).astype(np.uint8) * 255
    pred_mask_valid = pred_mask[pad_h:pad_h+new_h, pad_w:pad_w+new_w]
    binary = cv2.resize(pred_mask_valid, (w, h), interpolation=cv2.INTER_NEAREST)
    return binary

def estimate_physical_size(mask, depth_map_m, f_px):
    """
    通过伤口掩码和深度图，估算照片整体的物理宽度和高度。
    输入：
        mask: 二值掩码图，伤口区域为255，背景为0
        depth_map_m: 度量深度图，单位：米
        f_px: 焦距，单位：像素
    输出：
        width_mm, height_mm: 估算的物理宽度和高度，单位：毫米
    """
    mask_binary = (mask > 127).astype(np.uint8)
    ys, xs = np.where(mask_binary > 0)
    if len(xs) == 0:
        return None, None

    zs = depth_map_m[ys, xs]
    cx = mask.shape[1] / 2
    cy = mask.shape[0] / 2
    
    # 将伤口区域的像素坐标转换为真实物理坐标 (X, Y)
    X = (xs - cx) * zs / f_px
    Y = (ys - cy) * zs / f_px
    
    # 计算X和Y方向上的最大范围（即伤口的物理宽度和高度）
    width_m = np.max(X) - np.min(X)
    height_m = np.max(Y) - np.min(Y)
    
    # 转换为毫米
    width_mm = width_m * 1000
    height_mm = height_m * 1000
    
    # 合理性校验：皮肤照片的合理尺寸范围是30mm~300mm
    if not (30 <= width_mm <= 300 and 30 <= height_mm <= 300):
        return None, None
        
    return width_mm, height_mm