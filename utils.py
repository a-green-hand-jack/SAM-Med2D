from albumentations.pytorch import ToTensorV2
import cv2
import albumentations as A
import torch
import numpy as np
from torch.nn import functional as F
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import random
import torch.nn as nn
import logging
import os
from typing import Dict, Optional, Tuple, List


def get_boxes_from_mask(mask, box_num=1, std=0.1, max_pixel=5):
    """
    从给定的掩码图像中提取并扰动边界框。

    参数:
    - mask: 掩码图像，可以是torch.Tensor或numpy数组形式的二值掩码。
    - box_num: 需要提取的边界框数量，默认为1。
    - std: 扰动噪声的标准差，默认为0.1。
    - max_pixel: 扰动噪声的最大像素值，默认为5。

    返回:
    - noise_boxes: 扰动后的边界框列表，以torch.Tensor的形式返回。
    """
    # 如果掩码是torch.Tensor类型，则转换为numpy数组
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    # 使用label函数从掩码中提取连通区域的标签图像
    label_img = label(mask)
    # 使用regionprops函数获取所有连通区域的属性
    regions = regionprops(label_img)

    # 遍历所有连通区域并获取边界框坐标
    boxes = [tuple(region.bbox) for region in regions]

    # 如果生成的边界框数量大于类别数量，则按区域面积降序排序并选择前n个区域
    if len(boxes) >= box_num:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
        boxes = [tuple(region.bbox) for region in sorted_regions]

    # 如果生成的边界框数量小于类别数量，则复制现有边界框
    elif len(boxes) < box_num:
        num_duplicates = box_num - len(boxes)
        boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]

    # 对每个边界框进行噪声扰动
    noise_boxes = []
    for box in boxes:
        y0, x0, y1, x1 = box  # 边界框的坐标
        width, height = abs(x1 - x0), abs(y1 - y0)  # 边界框的宽度和高度
        # 计算扰动噪声的标准差和最大值
        noise_std = min(width, height) * std
        max_noise = min(max_pixel, int(noise_std * 5))
        # 为每个坐标添加随机噪声
        try:
            noise_x = np.random.randint(-max_noise, max_noise)
        except:
            noise_x = 0
        try:
            noise_y = np.random.randint(-max_noise, max_noise)
        except:
            noise_y = 0
        # 更新边界框坐标
        x0, y0 = x0 + noise_x, y0 + noise_y
        x1, y1 = x1 + noise_x, y1 + noise_y
        noise_boxes.append((x0, y0, x1, y1))

    # 将扰动后的边界框列表转换为torch.Tensor并返回
    return torch.as_tensor(noise_boxes, dtype=torch.float)


def select_random_points(
    pr: torch.Tensor, gt: torch.Tensor, point_num: int = 9
) -> Tuple[np.array, np.array]:
    """
    从预测掩码和真实掩码中随机选择点，并为这些点分配标签。

    参数:
    - pr: 预测掩码张量。
    - gt: 真实掩码张量。
    - point_num: 要选择的随机点的数量，默认为9。

    返回:
    - batch_points: 每个批次选择的点的坐标数组(x, y)。
    - batch_labels: 对应的标签数组（0代表背景，1代表前景）。
    """
    # 将预测掩码和真实掩码从GPU移动到CPU，并转换为NumPy数组
    pred, gt = pr.data.cpu().numpy(), gt.data.cpu().numpy()
    # 计算预测掩码和真实掩码不相等的点，作为误差图
    error = np.zeros_like(pred)
    error[pred != gt] = 1

    batch_points = []
    batch_labels = []
    # 遍历批次中的每个掩码
    for j in range(error.shape[0]):
        one_pred = pred[j].squeeze()  # 去除单维度
        one_gt = gt[j].squeeze()
        one_error = error[j].squeeze()

        # 获取误差图中非零点的索引
        indices = np.argwhere(one_error == 1)
        # 如果存在非零点，则从中随机选择point_num个点
        if indices.shape[0] > 0:
            selected_indices = indices[
                np.random.choice(indices.shape[0], point_num, replace=True)
            ]
        else:
            # 如果没有非零点，则在整个掩码上随机选择point_num个点
            indices = np.random.randint(0, one_error.shape[0], size=(point_num, 2))
            selected_indices = indices[
                np.random.choice(indices.shape[0], point_num, replace=True)
            ]
        selected_indices = selected_indices.reshape(-1, 2)

        # 初始化当前批次的点列表和标签列表
        points, labels = [], []
        for i in selected_indices:
            x, y = i  # 点的坐标
            # 根据预测掩码和真实掩码的值分配标签
            if one_pred[x, y] == 0 and one_gt[x, y] == 1:
                label = 1
            elif one_pred[x, y] == 1 and one_gt[x, y] == 0:
                label = 0
            else:
                label = -1
            # 将点坐标添加到列表中，并将y坐标取反以适应图像坐标系
            points.append((y, x))
            labels.append(label)

        # 将当前批次的点和标签列表添加到批次列表中
        batch_points.append(points)
        batch_labels.append(labels)

    # 将批次列表转换为NumPy数组并返回
    return np.array(batch_points), np.array(batch_labels)


def init_point_sampling(mask, get_point=1):
    """
    Initialization samples points from the mask and assigns labels to them.
    Args:
        mask (torch.Tensor): Input mask tensor.
        num_points (int): Number of points to sample. Default is 1.
    Returns:
        coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
        labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    # Get coordinates of black/white pixels
    fg_coords = np.argwhere(mask == 1)[:, ::-1]
    bg_coords = np.argwhere(mask == 0)[:, ::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor(
            [label], dtype=torch.int
        )
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = (
            torch.as_tensor(coords[indices], dtype=torch.float),
            torch.as_tensor(labels[indices], dtype=torch.int),
        )
        return coords, labels


def train_transforms(img_size, ori_h, ori_w):
    transforms = []
    if ori_h < img_size and ori_w < img_size:
        transforms.append(
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        )
    else:
        transforms.append(
            A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST)
        )
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms, p=1.0)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def generate_point(
    masks: torch.Tensor,
    labels: torch.Tensor,
    low_res_masks: torch.Tensor,
    batched_input: Dict[str, Optional[torch.Tensor]],
    point_num: int,
) -> Dict[str, Optional[torch.Tensor]]:
    """
    根据给定的掩码和标签，生成随机点坐标和对应的标签。

    参数:
    - masks: 高分辨率的掩码张量。
    - labels: 对应于掩码的标签张量。
    - low_res_masks: 低分辨率的掩码张量。
    - batched_input: 包含批处理数据的字典，如点坐标、标签等。
    - point_num: 要生成的点的数量。

    返回:
    - batched_input: 更新后的批处理输入字典，包含生成的点坐标和标签。
    """
    # 克隆掩码张量并将其通过sigmoid函数转换为概率图
    masks_clone = masks.clone()
    masks_sigmoid = torch.sigmoid(masks_clone)
    # 将概率大于0.5的位置设为1，其余设为0，得到二值掩码
    masks_binary = (masks_sigmoid > 0.5).float()

    # 克隆低分辨率掩码张量并应用sigmoid函数
    low_res_masks_clone = low_res_masks.clone()
    low_res_masks_logist = torch.sigmoid(low_res_masks_clone)

    # 根据二值掩码和标签选择随机点
    points, point_labels = select_random_points(
        pr=masks_binary, gt=labels, point_num=point_num
    )

    # 更新批处理输入字典，包含低分辨率掩码、点坐标、点标签和边界框
    batched_input["mask_inputs"] = low_res_masks_logist
    batched_input["point_coords"] = torch.as_tensor(points)
    batched_input["point_labels"] = torch.as_tensor(point_labels)
    batched_input["boxes"] = None

    return batched_input


def setting_prompt_none(batched_input):
    batched_input["point_coords"] = None
    batched_input["point_labels"] = None
    batched_input["boxes"] = None
    return batched_input


def draw_boxes(img, boxes):
    img_copy = np.copy(img)
    for box in boxes:
        cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return img_copy


def save_masks(
    preds,
    save_path,
    mask_name,
    image_size,
    original_size,
    pad=None,
    boxes=None,
    points=None,
    visual_prompt=False,
):
    ori_h, ori_w = original_size

    preds = torch.sigmoid(preds)
    preds[preds > 0.5] = int(1)
    preds[preds <= 0.5] = int(0)

    mask = preds.squeeze().cpu().numpy()
    mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    if visual_prompt:  # visualize the prompt
        if boxes is not None:
            boxes = boxes.squeeze().cpu().numpy()

            x0, y0, x1, y1 = boxes
            if pad is not None:
                x0_ori = int((x0 - pad[1]) + 0.5)
                y0_ori = int((y0 - pad[0]) + 0.5)
                x1_ori = int((x1 - pad[1]) + 0.5)
                y1_ori = int((y1 - pad[0]) + 0.5)
            else:
                x0_ori = int(x0 * ori_w / image_size)
                y0_ori = int(y0 * ori_h / image_size)
                x1_ori = int(x1 * ori_w / image_size)
                y1_ori = int(y1 * ori_h / image_size)

            boxes = [(x0_ori, y0_ori, x1_ori, y1_ori)]
            mask = draw_boxes(mask, boxes)

        if points is not None:
            point_coords, point_labels = (
                points[0].squeeze(0).cpu().numpy(),
                points[1].squeeze(0).cpu().numpy(),
            )
            point_coords = point_coords.tolist()
            if pad is not None:
                ori_points = [
                    [int((x * ori_w / image_size)), int((y * ori_h / image_size))]
                    if l == 0
                    else [x - pad[1], y - pad[0]]
                    for (x, y), l in zip(point_coords, point_labels)
                ]
            else:
                ori_points = [
                    [int((x * ori_w / image_size)), int((y * ori_h / image_size))]
                    for x, y in point_coords
                ]

            for point, label in zip(ori_points, point_labels):
                x, y = map(int, point)
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                mask[y, x] = color
                cv2.drawMarker(
                    mask,
                    (x, y),
                    color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=7,
                    thickness=2,
                )
    os.makedirs(save_path, exist_ok=True)
    mask_path = os.path.join(save_path, f"{mask_name}")
    cv2.imwrite(mask_path, np.uint8(mask))


# Loss funcation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p**self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_loss


class MaskIoULoss(nn.Module):
    def __init__(
        self,
    ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert (
            pred_mask.shape == ground_truth_mask.shape
        ), "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class FocalDiceloss_IoULoss(nn.Module):
    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss = self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_loss(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        return loss
