# file: predict_adaptive.py
# 基于 predict1109.py，在滑窗推理中使用自适应亮度增强。
#
# 与 predict1109.py 的区别：
#   - predict_with_sliding_window 中的预处理不再使用固定 brightness_offset=0.10
#   - 改为根据整幅图像的原始亮度均值动态调整偏移量（与 dataset_adaptive.py 一致）
#   - 亮图少增强或不增强，暗图正常增强

import os
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score

# ======================= 模型导入 =======================
from model0726 import DeepHybridCSNet

# ======================= 辅助函数 (保持不变) =======================
ID_TO_COLOR_MAP = {
    0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0),
    3: (255, 255, 0), 4: (0, 0, 255), 5: (255, 0, 255)
}
ID_TO_NAME_MAP = {
    0: 'background', 1: 'class_1', 2: 'class_2',
    3: 'class_AL', 4: 'class_black', 5: 'class_white'
}

def class_id_to_rgb(pred_map):
    """将类别ID图转换为可视化的RGB彩色图。"""
    rgb_map = np.zeros((*pred_map.shape, 3), dtype=np.uint8)
    for id, color in ID_TO_COLOR_MAP.items():
        rgb_map[pred_map == id] = color
    return rgb_map

def concatenate_pngs_in_folder(folder_path):
    """从文件夹加载并堆叠所有PNG图像为一个Numpy数组。"""
    arrays = []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    if not files: return None
    if len(files) > 5:
        files = np.concatenate((files[-5:], files[:-5])).tolist()
    for filename in files:
        try:
            with Image.open(os.path.join(folder_path, filename)) as img:
                arrays.append(np.array(img))
        except IOError as e:
            print(f"警告: 读取文件 {filename} 时出错: {e}")
    return np.stack(arrays, axis=-1) if arrays else None

def find_image_label_pairs(image_base_dir, label_base_dir, expected_bands=45):
    """扫描目录，只选择那些包含足够PNG文件且有对应标签的图像-标签对。"""
    pairs = []
    if not (os.path.isdir(image_base_dir) and os.path.isdir(label_base_dir)):
        print(f"错误：图像或标签根目录不存在！")
        return pairs
    image_subfolders = [d.name for d in os.scandir(image_base_dir) if d.is_dir()]
    for name in image_subfolders:
        image_folder_path = os.path.join(image_base_dir, name)
        if len([f for f in os.listdir(image_folder_path) if f.endswith('.png')]) == expected_bands:
            label_file_path = os.path.join(label_base_dir, f"{name}-90_mask.png")
            if os.path.isfile(label_file_path):
                pairs.append((image_folder_path, label_file_path))
    return pairs

# ======================= 评估函数 (已修改) =======================
def calculate_evaluation_metrics(prediction, ground_truth, num_classes, background_class_id=0):
    """
    【修改后】的评估函数:
    - 计算 mIoU (含背景), Kappa, AA。
    - 新增 Macro-Precision (仅前景)。
    - 移除了 OA。
    """
    metrics = {}
    foreground_classes = [c for c in range(num_classes) if c != background_class_id]
    all_classes = list(range(num_classes))
    non_background_mask = (ground_truth != background_class_id)

    if not np.any(non_background_mask):
        per_class_metrics = {cls: {'iou': np.nan, 'precision': np.nan, 'recall': np.nan} for cls in all_classes}
        return {'Kappa': 0, 'mIoU': 0, 'AA': 0, 'Macro-Precision': 0, 'per_class_metrics': per_class_metrics}
        
    pred_filtered = prediction[non_background_mask]
    true_filtered = ground_truth[non_background_mask]
    metrics['Kappa'] = cohen_kappa_score(true_filtered, pred_filtered, labels=foreground_classes)
    
    per_class_metrics = {}
    for cls in all_classes:
        pred_is_class = (prediction == cls)
        gt_is_class = (ground_truth == cls)
        tp = np.sum(pred_is_class & gt_is_class)
        fp = np.sum(pred_is_class & ~gt_is_class)
        fn = np.sum(~pred_is_class & gt_is_class)
        union = tp + fp + fn
        
        iou = tp / union if union > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        
        per_class_metrics[cls] = {'iou': iou, 'precision': precision, 'recall': recall}

    ious_for_miou = [per_class_metrics[c]['iou'] for c in all_classes if not np.isnan(per_class_metrics[c]['iou'])]
    metrics['mIoU'] = np.mean(ious_for_miou) if ious_for_miou else 0.0
    
    recalls_for_aa = [per_class_metrics[c]['recall'] for c in foreground_classes if not np.isnan(per_class_metrics[c]['recall'])]
    metrics['AA'] = np.mean(recalls_for_aa) if recalls_for_aa else 0.0

    precisions_for_ap = [per_class_metrics[c]['precision'] for c in foreground_classes if not np.isnan(per_class_metrics[c]['precision'])]
    metrics['Macro-Precision'] = np.mean(precisions_for_ap) if precisions_for_ap else 0.0

    metrics['per_class_metrics'] = per_class_metrics
    return metrics

# ======================= ★ 自适应增强预处理 =======================
def adaptive_enhance(hsi_image, brightness_offset=0.10, brightness_threshold=0.50):
    """
    与 dataset_adaptive.py 一致的预处理逻辑。
    
    1. 计算整幅图像原始亮度均值
    2. 逐波段 min-max 对比度拉伸
    3. 根据原始均值动态调整亮度偏移量
    
    hsi_image: (H, W, C) float32 numpy array, 原始 uint8 值 0-255
    返回: (H, W, C) float32, 值域 [0, 1]
    """
    hsi_float = hsi_image.astype(np.float32)
    
    # 计算原始亮度均值（归一化到 [0,1]）
    original_mean = float(hsi_float.mean()) / 255.0
    
    # 逐波段 min-max 对比度拉伸
    for c in range(hsi_float.shape[2]):
        band = hsi_float[:, :, c]
        b_min, b_max = band.min(), band.max()
        if b_max > b_min + 1e-6:
            hsi_float[:, :, c] = (band - b_min) / (b_max - b_min)
        else:
            hsi_float[:, :, c] = 0.0
    
    # 自适应亮度偏移
    brightness_factor = max(0.0, 1.0 - original_mean / brightness_threshold)
    adaptive_offset = brightness_offset * brightness_factor
    
    hsi_float = np.clip(hsi_float + adaptive_offset, 0, 1)
    return hsi_float


def predict_with_sliding_window(model, hsi_image, device, num_classes, window_size=(512, 512), stride=256,
                                brightness_offset=0.10, brightness_threshold=0.50):
    """
    使用滑窗对大尺寸高光谱图像进行预测。
    
    ★ 使用自适应增强预处理（与 dataset_adaptive.py 一致）。
    """
    model.eval()
    
    # ★ 自适应增强预处理
    hsi_image_norm = adaptive_enhance(
        hsi_image, 
        brightness_offset=brightness_offset,
        brightness_threshold=brightness_threshold
    )
    
    H, W, C = hsi_image_norm.shape
    win_h, win_w = window_size
    
    full_probs = np.zeros((H, W, num_classes), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y in tqdm(range(0, H, stride), desc="滑窗预测中", leave=False):
            for x in range(0, W, stride):
                y_end, x_end = min(y + win_h, H), min(x + win_w, W)
                patch = hsi_image_norm[y:y_end, x:x_end, :]
                
                tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device)
                
                output = model(tensor, target_size=(patch.shape[0], patch.shape[1]))
                probs = torch.softmax(output, dim=1).squeeze().cpu().numpy().transpose(1, 2, 0)
                
                full_probs[y:y_end, x:x_end] += probs
                count_map[y:y_end, x:x_end] += 1
                
    count_map[count_map == 0] = 1
    prediction_probs = full_probs / np.expand_dims(count_map, axis=-1)
    return np.argmax(prediction_probs, axis=2).astype(np.uint8)

# ======================= 主程序 =======================
if __name__ == '__main__':
    # ----- 用户配置 -----
    MODEL_PATH = r'your_path'
    IMAGE_BASE_DIR = 'your_path'
    LABEL_BASE_DIR = r'your_path'
    TEST_OUTPUT_DIR = 'your_path'
    
    NUM_CLASSES, NUM_BANDS = 6, 45
    BACKGROUND_CLASS_ID = 0
    WINDOW_SIZE, STRIDE = (256, 256), 128
    
    # ★ 自适应增强参数（需与训练时一致）
    BRIGHTNESS_OFFSET = 0.10
    BRIGHTNESS_THRESHOLD = 0.50
    
    # ----- 测试主程序 -----
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = DeepHybridCSNet(in_channels=NUM_BANDS, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("模型权重加载成功！")
    
    test_pairs = find_image_label_pairs(IMAGE_BASE_DIR, LABEL_BASE_DIR, expected_bands=NUM_BANDS)
    
    if not test_pairs:
        print("\n错误：未能加载任何符合规范的测试数据。")
    else:
        print(f"\n找到 {len(test_pairs)} 对有效的测试数据。开始评估...")
        all_metrics = []
        for img_folder_path, lbl_file_path in tqdm(test_pairs, desc="总体测试进度"):
            try:
                input_image = concatenate_pngs_in_folder(img_folder_path)
                if input_image is None: continue
                
                true_label = cv2.imread(lbl_file_path, cv2.IMREAD_GRAYSCALE)
                if true_label is None: continue
                
                prediction_map = predict_with_sliding_window(
                    model, input_image, device, NUM_CLASSES, WINDOW_SIZE, STRIDE,
                    brightness_offset=BRIGHTNESS_OFFSET,
                    brightness_threshold=BRIGHTNESS_THRESHOLD
                )
                
                prediction_bgr = cv2.cvtColor(class_id_to_rgb(prediction_map), cv2.COLOR_RGB2BGR)
                save_path = os.path.join(TEST_OUTPUT_DIR, f"prediction_{os.path.basename(img_folder_path)}.png")
                cv2.imwrite(save_path, prediction_bgr)
                
                metrics = calculate_evaluation_metrics(prediction_map.flatten(), true_label.flatten(), NUM_CLASSES, BACKGROUND_CLASS_ID)
                all_metrics.append(metrics)
            
            except Exception as e:
                print(f"\n处理 {os.path.basename(img_folder_path)} 时发生错误: {e}")
                continue
            
        # ----- 最终平均评估报告 -----
        if all_metrics:
            avg_miou = np.mean([m['mIoU'] for m in all_metrics])
            avg_precision = np.mean([m['Macro-Precision'] for m in all_metrics])
            avg_aa = np.mean([m['AA'] for m in all_metrics])
            avg_kappa = np.mean([m['Kappa'] for m in all_metrics])
            
            avg_per_class_metrics = {}
            for cls in range(NUM_CLASSES):
                valid_metrics = [m['per_class_metrics'][cls] for m in all_metrics]
                cls_ious = [vm['iou'] for vm in valid_metrics if not np.isnan(vm['iou'])]
                cls_precisions = [vm['precision'] for vm in valid_metrics if not np.isnan(vm['precision'])]
                cls_recalls = [vm['recall'] for vm in valid_metrics if not np.isnan(vm['recall'])]
                avg_per_class_metrics[cls] = {
                    'iou': np.mean(cls_ious) if cls_ious else 0.0,
                    'precision': np.mean(cls_precisions) if cls_precisions else 0.0,
                    'recall': np.mean(cls_recalls) if cls_recalls else 0.0,
                }
            
            print("\n\n" + "="*60)
            print(" " * 20 + "测试集总体评估报告")
            print("="*60)
            print(f"参与评估的图像总数: {len(all_metrics)}")
            print(f"平均交并比 (mIoU) [含背景]     : {avg_miou:.4f}")
            print(f"平均精度 (Macro-Precision) [仅前景]: {avg_precision:.4f}")
            print(f"平均召回率 (AA) [仅前景]      : {avg_aa:.4f}")
            print(f"平均Kappa系数 (Kappa) [仅前景]   : {avg_kappa:.4f}")
            print("-" * 60)
            print("--- 逐类别详细指标 (IoU, Precision, Recall) ---")
            header = f"{'类别':<25} | {'IoU':^10} | {'Precision':^10} | {'Recall':^10}"
            print(header)
            print("-" * len(header))
            for cls_id, cls_metrics in avg_per_class_metrics.items():
                class_name = ID_TO_NAME_MAP.get(cls_id, f"未知_{cls_id}")
                name_str = f"  - 类别 {cls_id} ({class_name})"
                print(f"{name_str:<25} | {cls_metrics['iou']:^10.4f} | {cls_metrics['precision']:^10.4f} | {cls_metrics['recall']:^10.4f}")
            print("="*60)
            print(f"所有预测图已保存到: {TEST_OUTPUT_DIR}")