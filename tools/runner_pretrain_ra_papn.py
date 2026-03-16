import torch
import torch.nn as nn
import time
import numpy as np
from tools import builder
from utils import misc, dist_utils
from utils.logger import *
from utils.AverageMeter import AverageMeter
import torch.nn.functional as F
from torchvision import transforms

# ===================== 1. 导入模型与增强 =====================
# 确保 models.PAPN 是您刚才整合好的那个单文件
from models.PAPN import PointPAPN
from datasets import data_transforms

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)

    # ===================== 2. 定义点云增强逻辑 =====================
    # 使用您指定的 ScaleAndTranslate
    view_augmentation = transforms.Compose([
        data_transforms.PointcloudScaleAndTranslate(),
    ])

    # 3. 构建数据集 (仅 Train)
    (train_sampler, train_dataloader), (_, _) = \
        builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)

    # 4. 构建模型 (PointPAPN)
    # 参数从 config.model 中读取，确保与 yaml 中的 key 对齐
    base_model = PointPAPN(
        in_channels=3,
        width=config.model.get('width', 32),
        blocks=config.model.get('blocks', [1, 1, 1, 1, 1]),
        strides=config.model.get('strides', [1, 2, 2, 2, 2, 1]),
        proj_dim=config.model.get('proj_dim', 256),
        n_parts=config.model.get('n_parts', 5),
        cra_alpha=config.model.get('cra_alpha', 1.0),
        last_stage_points=config.model.get('last_stage_points', 64),
        conv_args={} 
    )

    if args.use_gpu:
        base_model.to(args.local_rank)

    # 5. 分布式/断点续训设置
    start_epoch = 0
    if args.resume:
        start_epoch, _ = builder.resume_model(base_model, args, logger=logger)
    
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True
        )
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        base_model = nn.DataParallel(base_model).cuda()

    # 6. 构建优化器
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    # 7. 预训练主循环 (无验证版)
    base_model.zero_grad()
    
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        base_model.train()
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        losses = AverageMeter(['Total_Loss'])

        n_batches = len(train_dataloader)

        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            # 获取原始坐标 [B, N, 3]
            if config.dataset.train._base_.NAME == 'ShapeNet':
                points = data.cuda()
            else:
                points = data[0].cuda()

            # --- 核心修改：生成双视图 (im_q, im_k) ---
            # 针对同一份原始数据，调用两次随机增强
            points_q = view_augmentation(points.clone())
            points_k = view_augmentation(points.clone())

            optimizer.zero_grad()

            # --- 前向传播：计算模型内部聚合的 Loss ---
            # 直接返回：ssl_loss + sd_loss + cra_alpha * cra_loss
            loss = base_model(points_q, points_k)

            # --- 反向传播 ---
            loss.backward()
            optimizer.step()

            # 日志同步 (多卡)
            if args.distributed:
                loss_recorded = dist_utils.reduce_tensor(loss, args)
            else:
                loss_recorded = loss

            losses.update([loss_recorded.item()])
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] Loss = %.4f lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, 
                            losses.val()[0], optimizer.param_groups[0]['lr']), logger=logger)

        # 学习率更新
        if isinstance(scheduler, list):
            for item in scheduler: item.step(epoch)
        else:
            scheduler.step(epoch)

        epoch_end_time = time.time()
        print_log('[Training] EPOCH: %d Avg_Loss = %.4f Time = %.2fs' %
            (epoch, losses.avg()[0], epoch_end_time - epoch_start_time), logger=logger)

        # 定期保存模型 (由于没有验证，按 epoch 保存)
        if epoch % config.get('save_freq', 50) == 0:
            builder.save_checkpoint(base_model, optimizer, epoch, None, None, f'ckpt-epoch-{epoch}', args, logger=logger)
        
        # 始终保存最新的模型
        builder.save_checkpoint(base_model, optimizer, epoch, None, None, 'ckpt-last', args, logger=logger)      

    if train_writer is not None:
        train_writer.close()
