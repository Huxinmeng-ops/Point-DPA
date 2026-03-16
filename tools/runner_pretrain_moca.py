"""
Point-MOCA 预训练 Runner
改编自 runner_pretrain.py，添加 MOCA 特定的训练逻辑
"""
import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message


train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def compute_momentum_schedule(base_value, final_value, epochs, niter_per_ep, schedule='cosine'):
    """计算 momentum 调度"""
    momentum_schedule = np.ones(epochs * niter_per_ep) * final_value
    if schedule == 'cosine':
        for i in range(epochs):
            start_idx = i * niter_per_ep
            end_idx = (i + 1) * niter_per_ep
            momentum_schedule[start_idx:end_idx] = final_value - 0.5 * (final_value - base_value) * (
                1 + np.cos(np.pi * i / epochs))
    elif schedule == 'linear':
        for i in range(epochs):
            start_idx = i * niter_per_ep
            end_idx = (i + 1) * niter_per_ep
            momentum_schedule[start_idx:end_idx] = base_value + (final_value - base_value) * i / epochs
    else:  # constant
        momentum_schedule[:] = base_value
    
    return momentum_schedule


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    
    # 构建数据集
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                                builder.dataset_builder(args, config.dataset.val)
    
    # 构建模型
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    # DDP (必须在构建优化器之前)
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(
            base_model, device_ids=[args.local_rank % torch.cuda.device_count()], 
            find_unused_parameters=True
        )
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    
    # 优化器和调度器（必须在 DDP 包装之后）
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # 计算 momentum schedule
    niter_per_ep = len(train_dataloader)
    total_iters = config.max_epoch * niter_per_ep
    
    momentum_schedule_type = config.get('momentum_teacher_schedule', 'cosine')
    momentum_teacher = config.get('momentum_teacher', 0.996)
    momentum_teacher_end = config.get('momentum_teacher_end', 1.0)
    
    momentum_schedule = compute_momentum_schedule(
        momentum_teacher,
        momentum_teacher_end,
        config.max_epoch,
        niter_per_ep,
        schedule=momentum_schedule_type
    )
    
    print_log(f'[MOCA] Momentum schedule: {momentum_teacher} -> {momentum_teacher_end} ({momentum_schedule_type})', 
              logger=logger)
    
    # 训练循环
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'loss_img', 'loss_loc'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            global_step = epoch * niter_per_ep + idx
            
            # 获取当前的 momentum
            current_momentum = momentum_schedule[global_step]
            
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            
            # 数据预处理（与 runner_pretrain.py 保持一致）
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            
            assert points.size(1) == npoints
            points = train_transforms(points)
            
            # 前向传播
            # 传递训练参数
            loss = base_model(
                points, 
                momentum=current_momentum,
                img_weight=config.get('img_weight', 1.0),
                loc_weight=config.get('loc_weight', 1.0)
            )
            
            # 如果是 DDP，loss 已经是单个值
            if isinstance(loss, dict):
                loss_total = loss['loss']
                loss_img = loss.get('loss_img', 0)
                loss_loc = loss.get('loss_loc', 0)
            else:
                loss_total = loss
                loss_img = 0
                loss_loc = 0
            
            try:
                loss_total.backward()
            except:
                loss_total = loss_total.mean()
                loss_total.backward()

            # 梯度裁剪
            if config.get('grad_clip_norm', None) is not None:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_clip_norm)

            # 优化器步骤
            num_iter += 1
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss_total = dist_utils.reduce_tensor(loss_total, args)
                losses.update([loss_total.item() * 1000, 
                              loss_img.item() * 1000 if torch.is_tensor(loss_img) else loss_img * 1000,
                              loss_loc.item() * 1000 if torch.is_tensor(loss_loc) else loss_loc * 1000])
            else:
                losses.update([loss_total.item()* 1000, 
                              loss_img.item()* 1000  if torch.is_tensor(loss_img) else loss_img * 1000,
                              loss_loc.item() * 1000 if torch.is_tensor(loss_loc) else loss_loc * 1000])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss_total.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)
                train_writer.add_scalar('Momentum/Teacher', current_momentum, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %.6f (%.6f + %.6f) Momentum = %.4f lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            losses.val(0), losses.val(1), losses.val(2), current_momentum, optimizer.param_groups[0]['lr']), logger=logger)
        
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_img', losses.avg(1), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_loc', losses.avg(2), epoch)
        
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %.4f (%.4f + %.4f) lr = %.6f' %
            (epoch, epoch_end_time - epoch_start_time, losses.avg(0), losses.avg(1), losses.avg(2), optimizer.param_groups[0]['lr']), logger=logger)

        # 保存检查点
        if epoch % args.val_freq == 0 and epoch != 0:
            # 保存 ckpt
            if args.distributed:
                if dist_utils.get_rank() == 0:
                    save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
            else:
                save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
        
        # 定期保存
        if epoch % 25 == 0 and epoch >= 250:
            if args.distributed:
                if dist_utils.get_rank() == 0:
                    save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger=logger)
            else:
                save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger=logger)
    
    # 保存最终模型
    if args.distributed:
        if dist_utils.get_rank() == 0:
            save_checkpoint(base_model, optimizer, config.max_epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
    else:
        save_checkpoint(base_model, optimizer, config.max_epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
    
    print_log('Training finished!', logger=logger)


def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger=None):
    # 使用 builder.save_checkpoint 以保持一致性
    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger=logger)


def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=None):
    """验证（可选，Point-MOCA 主要关注预训练）"""
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode
    
    # 这里可以添加验证逻辑，比如：
    # 1. 提取特征后做最近邻分类
    # 2. 评估 codebook 的利用率
    # 3. 可视化重建结果
    
    # 目前返回一个虚拟 metric
    metrics = Acc_Metric(0.)
    return metrics


