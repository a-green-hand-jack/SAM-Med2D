from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn
import torch
import argparse
import os
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
import datetime
from torch.nn import functional as F
from apex import amp
import random

# debug
import debugpy

try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 7525))
    print("Waiting for debugger attach")
    print("the python code is train.py")
    print("the host is: localhost, the port is: 7525")
    debugpy.wait_for_client()
except Exception as e:
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument(
        "--run_name", type=str, default="sam-med2d", help="run model name"
    )
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=5, help="get mask number")
    parser.add_argument(
        "--data_path", type=str, default="data_demo", help="train data path"
    )
    parser.add_argument("--metrics", nargs="+", default=["iou", "dice"], help="metrics")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume")
    parser.add_argument(
        "--model_type", type=str, default="vit_b", help="sam model_type"
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="../pre-trained-model/SAM-Med/sam-med2d_b.pth",
        help="sam checkpoint",
    )
    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    parser.add_argument("--lr_scheduler", type=str, default=None, help="lr scheduler")
    parser.add_argument(
        "--point_list", type=list, default=[1, 3, 5, 9], help="point_list"
    )
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument(
        "--encoder_adapter", type=bool, default=True, help="use adapter"
    )
    parser.add_argument("--use_amp", type=bool, default=False, help="use amp")
    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key == "image" or key == "label":
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def prompt_and_decoder(
    args, batched_input, model, image_embeddings, decoder_iter=False
):
    """
    使用模型的编码器和解码器处理输入，生成掩码和IoU预测。

    参数:
    - args: 包含模型参数和配置的命名空间或字典。
    - batched_input: 包含点坐标、标签、边界框等信息的字典。
    - model: 包含编码器和解码器的深度学习模型。
    - image_embeddings: 输入图像的特征嵌入。
    - decoder_iter: 布尔值，指示是否在解码器迭代中使用无梯度上下文。

    返回:
    - masks: 高分辨率的掩码。
    - low_res_masks: 低分辨率的掩码。
    - iou_predictions: 预测的IoU值。
    """
    # 检查是否有点坐标，如果有，则提取点坐标和标签
    if batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    # 如果在解码器迭代中，使用无梯度上下文执行编码器
    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )
    else:
        # 否则，正常执行编码器
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

    # 使用编码器的输出和图像嵌入通过解码器生成低分辨率掩码和IoU预测
    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )

    # 如果启用多掩码输出，则选择最佳预测
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i : i + 1, idx])
        low_res_masks = torch.stack(low_res, 0)

    # 使用双线性插值将低分辨率掩码上采样到原始图像大小
    masks = F.interpolate(
        low_res_masks,
        size=(args.image_size, args.image_size),
        mode="bilinear",
        align_corners=False,
    )

    return masks, low_res_masks, iou_predictions


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion):
    """
    Trains the model for one epoch.

    Parameters:
    - args: Namespace containing configuration arguments.
    - model: The neural network model to be trained.
    - optimizer: The optimizer used for updating model parameters.
    - train_loader: DataLoader for the training dataset.
    - epoch: The current epoch number.
    - criterion: The loss function used for training.

    Returns:
    - train_losses: List of loss values for each batch.
    - train_iter_metrics: List of metrics calculated for each iteration.
    """
    # 使用tqdm库来显示训练进度条
    train_loader = tqdm(train_loader)
    train_losses = []  # 存储每个batch的损失值
    train_iter_metrics = [0] * len(args.metrics)  # 初始化迭代指标

    # 遍历训练数据集
    for batch, batched_input in enumerate(train_loader):
        # 将输入数据堆叠到一个字典中，并移动到指定的设备上
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)

        # 随机选择是否使用点坐标或边界框作为输入
        if random.random() > 0.5:
            batched_input["point_coords"] = None
            flag = "boxes"
        else:
            batched_input["boxes"] = None
            flag = "point"

        # 设置模型参数的梯度跟踪
        for n, value in model.image_encoder.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

        # 如果使用自动混合精度(AMP)，则将标签和图像嵌入转换为半精度
        if args.use_amp:  # 默认为False
            labels = batched_input["label"].half()
            image_embeddings = model.image_encoder(batched_input["image"].half())
        else:
            labels = batched_input["label"]
            image_embeddings = model.image_encoder(batched_input["image"])

        # 重复图像嵌入以匹配mask的数量
        B, _, _, _ = image_embeddings.shape
        image_embeddings_repeat = []
        for i in range(B):
            image_embed = image_embeddings[i].repeat(args.mask_num, 1, 1, 1)
            image_embeddings_repeat.append(image_embed)
        image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

        # 调用prompt_and_decoder函数进行模型的前向传播和损失计算
        masks, low_res_masks, iou_predictions = prompt_and_decoder(
            args, batched_input, model, image_embeddings, decoder_iter=False
        )
        loss = criterion(masks, labels, iou_predictions)

        # 根据是否使用AMP来执行反向传播
        if args.use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=False)
        else:
            loss.backward(retain_graph=False)

        # 更新模型参数并重置梯度
        optimizer.step()
        optimizer.zero_grad()

        # 每50个batch打印一次训练指标
        if (batch + 1) % 50 == 0:
            print(
                f"Epoch: {epoch+1}, Batch: {batch+1}, first {flag} prompt: {SegMetrics(masks, labels, args.metrics)}"
            )

        # 随机选择点的数量，并生成新的输入数据
        point_num = random.choice(args.point_list)  # 默认[1, 3, 5, 9]
        batched_input = generate_point(
            masks, labels, low_res_masks, batched_input, point_num
        )
        batched_input = to_device(batched_input, args.device)

        # 为后续迭代设置模型参数的梯度跟踪
        image_embeddings = image_embeddings.detach().clone()
        for n, value in model.named_parameters():
            if "image_encoder" in n:
                value.requires_grad = False
            else:
                value.requires_grad = True

        # 随机选择初始mask数量，并进行多次迭代
        init_mask_num = np.random.randint(1, args.iter_point - 1)  # 默认8
        for iter in range(args.iter_point):
            # 根据迭代次数设置prompt为None
            if iter == init_mask_num or iter == args.iter_point - 1:
                batched_input = setting_prompt_none(batched_input)

            # 进行模型的前向传播和损失计算
            # 和第一次的前向传播过程相比，输入的batched_input中point_coords、point_labels和boxes的值被替换为None
            masks, low_res_masks, iou_predictions = prompt_and_decoder(
                args, batched_input, model, image_embeddings, decoder_iter=True
            )
            loss = criterion(masks, labels, iou_predictions)

            # 执行反向传播
            if args.use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
            else:
                loss.backward(retain_graph=True)

            # 更新模型参数并重置梯度
            optimizer.step()
            optimizer.zero_grad()

            # 每50个batch打印一次训练指标
            if (batch + 1) % 50 == 0:
                if iter == init_mask_num or iter == args.iter_point - 1:
                    print(
                        f"Epoch: {epoch+1}, Batch: {batch+1}, mask prompt: {SegMetrics(masks, labels, args.metrics)}"
                    )
                else:
                    print(
                        f"Epoch: {epoch+1}, Batch: {batch+1}, point {point_num} prompt: {SegMetrics(masks, labels, args.metrics)}"
                    )

        # 每200个batch保存一次模型状态
        if (batch + 1) % 200 == 0:
            print(f"epoch:{epoch+1}, iteration:{batch+1}, loss:{loss.item()}")
            save_path = os.path.join(
                f"{args.work_dir}/models",
                args.run_name,
                f"epoch{epoch+1}_batch{batch+1}_sam.pth",
            )
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(state, save_path)

        # 记录每个batch的损失值
        train_losses.append(loss.item())

        # 更新tqdm进度条的信息
        gpu_info = {"gpu_name": args.device}
        train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)

        # 计算并累积迭代指标
        train_batch_metrics = SegMetrics(masks, labels, args.metrics)
        train_iter_metrics = [
            train_iter_metrics[i] + train_batch_metrics[i]
            for i in range(len(args.metrics))
        ]

    # 返回损失值列表和迭代指标列表
    return train_losses, train_iter_metrics


def main(args):
    model = sam_model_registry[args.model_type](args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalDiceloss_IoULoss()

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[5, 10], gamma=0.5
        )
        print("*******Use MultiStepLR")

    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"].state_dict())
            print(f"*******load {args.resume}")

    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        print("*******Mixed precision with Apex")
    else:
        print("*******Do not use mixed precision")

    train_dataset = TrainingDataset(
        args.data_path,
        image_size=args.image_size,
        mode="train",
        point_num=1,
        mask_num=args.mask_num,
        requires_name=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )  # default 2
    print("*******Train data:", len(train_dataset))

    loggers = get_logger(
        os.path.join(
            args.work_dir,
            "logs",
            f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}",
        )
    )

    best_loss = 1e10
    len_dataloader = len(train_loader)

    for epoch in range(0, args.epochs):
        model.train()
        train_metrics = {}
        start = time.time()
        os.makedirs(
            os.path.join(f"{args.work_dir}/models", args.run_name), exist_ok=True
        )
        train_losses, train_iter_metrics = train_one_epoch(
            args, model, optimizer, train_loader, epoch, criterion
        )

        if args.lr_scheduler is not None:
            scheduler.step()

        train_iter_metrics = [metric / len_dataloader for metric in train_iter_metrics]
        train_metrics = {
            args.metrics[i]: "{:.4f}".format(train_iter_metrics[i])
            for i in range(len(train_iter_metrics))
        }

        average_loss = np.mean(train_losses)
        lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr
        loggers.info(
            f"epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics}"
        )

        if average_loss < best_loss:
            best_loss = average_loss
            save_path = os.path.join(
                args.work_dir, "models", args.run_name, f"epoch{epoch+1}_sam.pth"
            )
            state = {"model": model.float().state_dict(), "optimizer": optimizer}
            torch.save(state, save_path)
            if args.use_amp:
                model = model.half()

        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))


if __name__ == "__main__":
    args = parse_args()
    main(args)
