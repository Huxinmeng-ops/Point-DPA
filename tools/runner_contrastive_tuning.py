import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from datasets import data_transforms
from torchvision import transforms
import numpy as np

# Data augmentations for contrastive learning on point clouds
train_transforms_weak = transforms.Compose([
    data_transforms.PointcloudScaleAndTranslate(),
])

train_transforms_strong = transforms.Compose([
    data_transforms.PointcloudScale(),
    data_transforms.PointcloudRotate(),
    data_transforms.PointcloudTranslate(),
    data_transforms.PointcloudJitter(),
])


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


def get_layer_wise_lr_decay_params(model, lr, layer_decay_rate=0.65, num_layers=12):
    """
    Apply layer-wise learning rate decay
    Lower layers get smaller learning rates
    """
    parameter_group_names = {}
    parameter_group_vars = {}
    
    # Get the base model (unwrap DDP/DataParallel if needed)
    if hasattr(model, 'module'):
        base_model = model.module
    else:
        base_model = model
    
    # Freeze bottom half of transformer blocks
    freeze_depth = num_layers // 2
    
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Determine layer index for transformer blocks
        layer_id = None
        if 'blocks.' in name:
            # Extract block index
            try:
                block_idx = int(name.split('blocks.')[1].split('.')[0])
                layer_id = block_idx
            except:
                layer_id = None
        
        # Freeze bottom half
        if layer_id is not None and layer_id < freeze_depth:
            param.requires_grad = False
            continue
        
        # Compute learning rate with layer-wise decay
        if layer_id is not None:
            # Apply decay from top to bottom
            decay_factor = layer_decay_rate ** (num_layers - layer_id - 1)
            group_name = f"layer_{layer_id}"
        else:
            # Other parameters (projector, predictor, etc.) use full lr
            decay_factor = 1.0
            group_name = "no_decay"
        
        current_lr = lr * decay_factor
        
        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "params": [],
                "lr": current_lr,
                "name": group_name,
            }
            parameter_group_vars[group_name] = {
                "params": [],
                "lr": current_lr,
                "name": group_name,
            }
        
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    
    print_log(f"Param groups with layer-wise LR decay:", logger='ContrastiveTuning')
    for group_name, group in parameter_group_names.items():
        print_log(f"  {group_name}: lr={parameter_group_vars[group_name]['lr']:.6f}, "
                 f"num_params={len(group['params'])}", logger='ContrastiveTuning')
    
    return list(parameter_group_vars.values())


def freeze_bottom_layers(model, freeze_ratio=0.5):
    """Freeze bottom layers of the transformer encoder"""
    if hasattr(model, 'module'):
        base_model = model.module
    else:
        base_model = model
    
    if hasattr(base_model, 'blocks'):
        num_blocks = len(base_model.blocks)
        freeze_depth = int(num_blocks * freeze_ratio)
        
        for idx, block in enumerate(base_model.blocks):
            if idx < freeze_depth:
                for param in block.parameters():
                    param.requires_grad = False
        
        print_log(f"Frozen bottom {freeze_depth}/{num_blocks} transformer blocks", logger='ContrastiveTuning')
    
    # Also freeze the initial encoder and pos_embed
    if hasattr(base_model, 'encoder'):
        for param in base_model.encoder.parameters():
            param.requires_grad = False
        print_log(f"Frozen point encoder", logger='ContrastiveTuning')
    
    if hasattr(base_model, 'pos_embed'):
        for param in base_model.pos_embed.parameters():
            param.requires_grad = False
        print_log(f"Frozen positional embedding", logger='ContrastiveTuning')


def train_epoch(base_model, train_loader, optimizer, epoch, config, logger, train_writer):
    """Train one epoch for contrastive tuning"""
    base_model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(['Loss'])
    
    num_iter = len(train_loader)
    base_model.zero_grad()
    
    n_batches = len(train_loader)
    end = time.time()
    
    # Get max_epoch value (handle both attribute and dict access)
    max_epoch = config.max_epoch if hasattr(config, 'max_epoch') else config.get('max_epoch', 100)
    
    for idx, (taxonomy_ids, model_ids, data) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        # Extract points and labels from data tuple
        points = data[0] if isinstance(data, tuple) else data
        
        # Generate two augmented views
        points1 = points.cuda(non_blocking=True)
        points2 = points.clone().cuda(non_blocking=True)
        
        # Apply augmentations
        use_strong_aug = config.get('use_strong_aug', False)
        if use_strong_aug:
            points1 = train_transforms_strong(points1)
            points2 = train_transforms_strong(points2)
        else:
            points1 = train_transforms_weak(points1)
            points2 = train_transforms_weak(points2)
        
        # Forward pass
        loss = base_model(points1, points2)
        
        if isinstance(loss, dict):
            loss = loss['loss']
        
        loss.backward()
        
        # Gradient accumulation
        if (idx + 1) % config.step_per_update == 0:
            # Gradient clipping
            if config.get('grad_clip_norm', None):
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_clip_norm)
            
            optimizer.step()
            base_model.zero_grad()
        
        # Update metrics
        losses.update([loss.item()])
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Logging
        if idx % config.get('print_freq', 10) == 0:
            print_log(
                f'Epoch [{epoch}/{max_epoch}][{idx}/{n_batches}]\t'
                f'Time {batch_time.val():.3f} ({batch_time.avg():.3f})\t'
                f'Data {data_time.val():.3f} ({data_time.avg():.3f})\t'
                f'Loss {losses.val(0):.4f} ({losses.avg(0):.4f})',
                logger=logger
            )
        
        if train_writer is not None:
            train_writer.add_scalar('Loss/Batch', losses.val(0), 
                                   epoch * num_iter + idx)
    
    if train_writer is not None:
        train_writer.add_scalar('Loss/Epoch', losses.avg(0), epoch)
    
    print_log(f'Train Epoch [{epoch}/{max_epoch}] Loss: {losses.avg(0):.4f}', logger=logger)
    
    return losses.avg(0)


def validate_linear_probe(base_model, train_loader, test_loader, epoch, logger):
    """
    Validate the representation quality using linear probing
    Train a linear classifier on frozen features
    """
    from sklearn.linear_model import LogisticRegression
    
    base_model.eval()
    
    # Extract features from training set
    print_log('Extracting training features...', logger=logger)
    train_features = []
    train_labels = []
    
    with torch.no_grad():
        for taxonomy_ids, model_ids, data in train_loader:
            # Extract points and labels
            if isinstance(data, tuple):
                points, labels = data
            else:
                points = data
                labels = taxonomy_ids  # fallback
            
            points = points.cuda(non_blocking=True)
            # Get representation
            if hasattr(base_model, 'module'):
                feats = base_model.module.forward_encoder(points)
            else:
                feats = base_model.forward_encoder(points)
            
            train_features.append(feats.cpu().numpy())
            if isinstance(labels, torch.Tensor):
                train_labels.append(labels.numpy())
            else:
                train_labels.append(np.array(labels))
    
    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    
    # Extract features from test set
    print_log('Extracting test features...', logger=logger)
    test_features = []
    test_labels = []
    
    with torch.no_grad():
        for taxonomy_ids, model_ids, data in test_loader:
            # Extract points and labels
            if isinstance(data, tuple):
                points, labels = data
            else:
                points = data
                labels = taxonomy_ids  # fallback
            
            points = points.cuda(non_blocking=True)
            # Get representation
            if hasattr(base_model, 'module'):
                feats = base_model.module.forward_encoder(points)
            else:
                feats = base_model.forward_encoder(points)
            
            test_features.append(feats.cpu().numpy())
            if isinstance(labels, torch.Tensor):
                test_labels.append(labels.numpy())
            else:
                test_labels.append(np.array(labels))
    
    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    # Train linear classifier
    print_log('Training linear classifier...', logger=logger)
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(train_features, train_labels)
    
    # Evaluate
    acc = clf.score(test_features, test_labels) * 100
    print_log(f'Epoch [{epoch}] Linear Probe Accuracy: {acc:.2f}%', logger=logger)
    
    return acc


def run_net(args, config, train_writer=None, val_writer=None):
    """Main function for contrastive tuning training"""
    logger = get_logger(args.log_name)
    
    # Build dataset
    print_log('Building datasets...', logger=logger)
    (train_sampler, train_dataloader), (_, test_dataloader) = \
        builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    
    # Build model
    print_log('Building Point-MAE-CT model...', logger=logger)
    base_model = builder.model_builder(config.model)
    
    # Load pretrained Point-MAE checkpoint
    if hasattr(config, 'pretrained_ckpt') and config.pretrained_ckpt:
        print_log(f'Loading pretrained Point-MAE from {config.pretrained_ckpt}', logger=logger)
        if hasattr(base_model, 'load_model_from_ckpt'):
            base_model.load_model_from_ckpt(config.pretrained_ckpt)
        else:
            builder.load_model(base_model, config.pretrained_ckpt, logger=logger)
    
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    # Freeze settings based on stage
    stage = config.get('stage', 'init')  # 'init' or 'tune'
    
    if stage == 'init':
        # Stage 1: NNCLR initialization with frozen encoder
        print_log('Stage 1: NNCLR Initialization (Encoder Frozen)', logger=logger)
        if hasattr(base_model, 'module'):
            base_model = base_model.module
        
        # Freeze entire encoder
        for name, param in base_model.named_parameters():
            if 'projector' not in name and 'predictor' not in name and 'queue' not in name:
                param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in base_model.parameters())
        print_log(f'Trainable params: {trainable_params:,} / {total_params:,} '
                 f'({100.0 * trainable_params / total_params:.2f}%)', logger=logger)
    
    # Parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    
    # Resume checkpoint
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metric)
    
    # DDP
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(
            base_model, 
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True
        )
        print_log('Using Distributed Data Parallel...', logger=logger)
    else:
        print_log('Using Data Parallel...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    
    # Optimizer & scheduler
    if stage == 'tune':
        # Stage 2: Contrastive Tuning with layer-wise LR decay
        print_log('Stage 2: Contrastive Tuning (Layer-wise LR Decay)', logger=logger)
        
        # Apply layer-wise learning rate decay
        num_layers = config.model.transformer_config.depth
        layer_decay_rate = config.get('layer_decay_rate', 0.65)
        param_groups = get_layer_wise_lr_decay_params(
            base_model, 
            config.optimizer.kwargs.lr,
            layer_decay_rate=layer_decay_rate,
            num_layers=num_layers
        )
        
        # Create optimizer with layer-wise parameters
        if config.optimizer.type == 'AdamW':
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=config.optimizer.kwargs.get('weight_decay', 0.05)
            )
        else:
            optimizer = builder.build_opti_sche(base_model, config)[0]
    else:
        # Standard optimizer for init stage
        optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    # Build scheduler
    if stage == 'tune':
        if config.scheduler.type == 'CosLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.scheduler.kwargs.epochs,
                eta_min=1e-6
            )
        else:
            scheduler = builder.build_opti_sche(base_model, config)[1]
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)
    
    # Training loop
    print_log('Start training...', logger=logger)
    base_model.zero_grad()
    
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Train one epoch
        train_loss = train_epoch(
            base_model, train_dataloader, optimizer, 
            epoch, config, logger, train_writer
        )
        
        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step(epoch)
            else:
                scheduler.step(epoch)
        
        # Save checkpoint
        if epoch % config.get('save_freq', 10) == 0 or epoch == config.max_epoch:
            if args.distributed:
                if args.local_rank == 0:
                    builder.save_checkpoint(
                        base_model, optimizer, epoch, Acc_Metric(train_loss), best_metrics,
                        'ckpt-last', args, logger=logger
                    )
            else:
                builder.save_checkpoint(
                    base_model, optimizer, epoch, Acc_Metric(train_loss), best_metrics,
                    f'ckpt-epoch-{epoch}', args, logger=logger
                )
        
        # Optional: Linear probe validation (expensive, run less frequently)
        if config.get('validate_freq', None) and epoch % config.validate_freq == 0:
            if args.local_rank == 0:
                acc = validate_linear_probe(
                    base_model, train_dataloader, test_dataloader,
                    epoch, logger
                )
                
                if val_writer is not None:
                    val_writer.add_scalar('LinearProbe/Acc', acc, epoch)
                
                # Save best model
                current_metric = Acc_Metric(acc)
                if current_metric.better_than(best_metrics):
                    best_metrics = current_metric
                    builder.save_checkpoint(
                        base_model, optimizer, epoch, current_metric, best_metrics,
                        'ckpt-best', args, logger=logger
                    )
    
    # Save final model
    if args.local_rank == 0:
        builder.save_checkpoint(
            base_model, optimizer, config.max_epoch, best_metrics, best_metrics,
            'ckpt-final', args, logger=logger
        )
    
    print_log('Training completed!', logger=logger)

