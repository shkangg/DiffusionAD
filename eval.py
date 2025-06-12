import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc, roc_auc_score, average_precision_score, roc_curve
from skimage.measure import label, regionprops
from tqdm import tqdm
from math import exp
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils

# ✅ pt 기반 데이터셋
class PreprocessedVideoDataset_fortest(Dataset):
    def __init__(self, pt_dir):
        self.pt_files = sorted([
            os.path.join(pt_dir, f) for f in os.listdir(pt_dir)
            if f.endswith(".pt")
        ])

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_path = self.pt_files[idx]
        data = torch.load(pt_path)
        frames = data["frames"].permute(0, 3, 1, 2)  # (C, T, H, W) → (T, C, H, W)

        filename = os.path.basename(pt_path).lower()
        if "fi" in filename:
            label = 1
        elif "no" in filename:
            label = 0
        else:
            raise ValueError(f"Unknown label in file: {filename}")

        return frames, label

# ✅ 유틸 함수들
def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min + 1e-8)

def cvt2heatmap(gray):
    return cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)

def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def pixel_pro(mask, pred):
    mask = np.asarray(mask, dtype=bool)
    pred = np.asarray(pred)
    max_step = 1000
    expect_fpr = 0.3
    max_th = pred.max()
    min_th = pred.min()
    delta = (max_th - min_th) / max_step
    ious_mean, pros_mean, fprs = [], [], []
    binary_score_maps = np.zeros_like(pred, dtype=bool)

    for step in range(max_step):
        thred = max_th - step * delta
        binary_score_maps[pred <= thred] = 0
        binary_score_maps[pred > thred] = 1
        pro, iou = [], []

        for i in range(len(binary_score_maps)):
            label_map = label(mask[i], connectivity=2)
            props = regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)

            intersection = np.logical_and(binary_score_maps[i], mask[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], mask[i]).astype(np.float32).sum()
            if mask[i].any():
                iou.append(intersection / union)

        ious_mean.append(np.mean(iou))
        pros_mean.append(np.mean(pro))
        fpr = np.logical_and(~mask, binary_score_maps).sum() / (~mask).sum()
        fprs.append(fpr)

    fprs, pros_mean = np.array(fprs), np.array(pros_mean)
    idx = fprs <= expect_fpr
    fprs_selected = min_max_norm(fprs[idx])
    pros_mean_selected = pros_mean[idx]
    return auc(fprs_selected, pros_mean_selected)

def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd

def load_checkpoint(param, device, sub_class, checkpoint_type, args):
    ck_path = f'{args["output_path"]}/model/diff-params-ARGS={param}/{sub_class}/params-{checkpoint_type}.pt'
    return torch.load(ck_path, map_location=device)

def load_parameters(device, sub_class, checkpoint_type):
    param = "args1.json"
    with open(f'./args/{param}', 'r') as f:
        args = json.load(f)
    args['arg_num'] = param[4:-5]
    args = defaultdict_from_json(args)
    output = load_checkpoint(param[4:-5], device, sub_class, checkpoint_type, args)
    return args, output

# ✅ 핵심 테스트 루프
def testing(test_loader, args, unet_model, seg_model, data_len, sub_class, class_type, checkpoint_type, device):
    from models.DDPM import GaussianDiffusionModel, get_beta_schedule

    normal_t = args["eval_normal_t"]
    noiser_t = args["eval_noisier_t"]

    os.makedirs(f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}', exist_ok=True)

    in_channels = args["channels"]
    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    ddpm_sample = GaussianDiffusionModel(
        args['img_size'], betas,
        loss_weight=args['loss_weight'],
        loss_type=args['loss-type'],
        noise=args["noise_fn"],
        img_channels=in_channels
    )

    total_image_pred = []
    total_image_gt = []
    total_pixel_gt = []
    total_pixel_pred = []
    gt_matrix_pixel = []
    pred_matrix_pixel = []

    for i, (frames, label) in enumerate(tqdm(test_loader)):
        frames = frames.cuda()  # [B, T, C, H, W]
        label = label.cuda()

        B, T, C, H, W = frames.size()
        real_rgb = frames[:, -6:, :, :, :]  # 마지막 6프레임 → RGB
        input_image = real_rgb[:, 0]  # 첫 프레임만 사용

        normal_t_tensor = torch.tensor([normal_t], device=device).repeat(B)
        noiser_t_tensor = torch.tensor([noiser_t], device=device).repeat(B)

        with torch.no_grad():
            loss, pred_x_0_condition, pred_x_0_normal, pred_x_0_noisier, x_normal_t, x_noiser_t, pred_x_t_noisier = \
                ddpm_sample.norm_guided_one_step_denoising_eval(unet_model, input_image, normal_t_tensor, noiser_t_tensor, args)

            pred_mask = seg_model(torch.cat((input_image, pred_x_0_condition), dim=1))

        out_mask = pred_mask

        gt_mask = torch.zeros_like(out_mask).cuda()
        if label.item() == 1:
            gt_mask[:, :, 32:96, 32:96] = 1.0  # 예시 마스크 (수정 가능)

        gt_matrix_pixel.extend(gt_mask[0].detach().cpu().numpy().astype(int))
        pred_matrix_pixel.extend(out_mask[0].detach().cpu().numpy())

        image_score = torch.mean(torch.topk(out_mask[0].flatten(), 50).values)
        total_image_pred.append(image_score.item())
        total_image_gt.append(label.item())

        total_pixel_pred.extend(out_mask[0].flatten().detach().cpu().numpy())
        total_pixel_gt.extend(gt_mask[0].flatten().detach().cpu().numpy())

    # 성능 출력
    auroc_image = round(roc_auc_score(total_image_gt, total_image_pred), 3) * 100
    auroc_pixel = round(roc_auc_score(total_pixel_gt, total_pixel_pred), 3) * 100
    ap_pixel = round(average_precision_score(total_pixel_gt, total_pixel_pred), 3) * 100
    aupro_pixel = round(pixel_pro(gt_matrix_pixel, pred_matrix_pixel), 3) * 100

    print("Image AUC-ROC:", auroc_image)
    print("Pixel AUC-ROC:", auroc_pixel)
    print("Pixel-AP:", ap_pixel)
    print("Pixel-AUPRO:", aupro_pixel)

    df = {
        "classname": [sub_class],
        "Image-AUROC": [auroc_image],
        "Pixel-AUROC": [auroc_pixel],
        "Pixel-AUPRO": [aupro_pixel],
        "Pixel_AP": [ap_pixel]
    }
    import pandas as pd
    df_class = pd.DataFrame(df)
    csv_path = f"{args['output_path']}/metrics/ARGS={args['arg_num']}/{normal_t}_{noiser_t}t_{class_type}_{args['condition_w']}condition_{checkpoint_type}ck.csv"
    df_class.to_csv(csv_path, mode='a', header=False, index=False)

# ✅ 메인 실행부
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sub_class = "final_test"
    checkpoint_type = 'best'
    pt_test_dir = "./pt_data/final_test"

    args, output = load_parameters(device, sub_class, checkpoint_type)

    from models.Recon_subnetwork import UNetModel
    from models.Seg_subnetwork import SegmentationSubNetwork

    unet_model = UNetModel(
        args['img_size'][0], args['base_channels'],
        channel_mults=args['channel_mults'], dropout=args["dropout"],
        n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
        in_channels=args["channels"]
    ).to(device)

    seg_model = SegmentationSubNetwork(in_channels=6, out_channels=1).to(device)

    unet_model.load_state_dict(output["unet_model_state_dict"])
    seg_model.load_state_dict(output["seg_model_state_dict"])
    unet_model.eval()
    seg_model.eval()

    test_dataset = PreprocessedVideoDataset_fortest(pt_test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    data_len = len(test_dataset)

    testing(test_loader, args, unet_model, seg_model, data_len, sub_class, class_type="PT", checkpoint_type=checkpoint_type, device=device)

if __name__ == "__main__":
    main()
