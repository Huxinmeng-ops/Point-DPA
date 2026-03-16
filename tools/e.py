import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import random

# ==========================================
# 1. CRA (Channel-wise Region Alignment) 模块
# ==========================================
class MCRegionLoss(nn.Module):
    def __init__(self, num_classes=196, cnums=5, cgroups=[196], p=0.6, feat_dim=2048):
        """
        CRA 模块: 强制特征通道关注特定空间区域
        num_classes: 空间区域总数 (H*W), ResNet50 layer4 通常为 14*14=196
        """
        super().__init__()
        # 针对 ResNet50 layer4 (2048 channels) 的优化分组逻辑
        if num_classes == 196 and feat_dim == 2048:
            cgroups = [108, 88]
            cnums = [10, 11]
            
        self.cnums = cnums
        self.cgroups = cgroups
        self.p = p
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, feat):
        """
        feat: Backbone 输出的特征图 [B, C, H, W]
        """
        n, c, h, w = feat.size()
        
        # 生成区域标签
        label_np = []
        for i in range(len(self.cgroups)):
            if i > 0:
                label_np.append(np.repeat(np.arange(self.cgroups[i]), self.cnums[i]) + self.cgroups[i - 1])
            else:
                label_np.append(np.repeat(np.arange(self.cgroups[i]), self.cnums[i]))
        label = torch.from_numpy(np.concatenate(label_np)).to(feat.device)
        
        # 通道分组与排列
        sp = [0]
        tmp = np.array(self.cgroups) * np.array(self.cnums)
        for i in range(len(self.cgroups)):
            sp.append(sum(tmp[: i + 1]))

        dis_branch = []
        for i in range(1, len(sp)):
            # 提取对应通道组并展平空间维度
            feat_group = feat[:, sp[i-1] : sp[i]]
            feat_group = feat_group.view(n, -1, h * w) # [B, C_group, H*W]
            dis_branch.append(feat_group)

        # 拼接所有通道分支
        dis_branch = torch.cat(dis_branch, dim=1) # [B, C, H*W]
        
        # 计算 CRA 损失: 对每个 Batch 中的图像，计算通道对区域的分类准确性
        l_dis = torch.stack([self.celoss(dis_branch[d, :, :], label) for d in range(n)]).mean()

        return l_dis

# ==========================================
# 2. 修改后的 PAPN Encoder (支持返回特征图)
# ==========================================
layer_dims = {
    'layer1': 256,
    'layer2': 512,
    'layer3': 1024,
    'layer4': 2048
}

class Encoder(nn.Module):
    def __init__(self, base_encoder, proj_dim, layer_proj_dim, layer_names, pretrain):
        super().__init__()
        self.num_layers = len(layer_names)
        self.encoder = timm.create_model(base_encoder, pretrained=False, num_classes=0)

        if pretrain is not None:
            checkpoint = torch.load(pretrain, map_location="cpu")
            for key in list(checkpoint.keys()):
                if "fc" in key: del checkpoint[key]
            self.encoder.load_state_dict(checkpoint, strict=True)

        self.layers_fc = nn.ModuleList([])
        self.layers_emb = []

        dims = [layer_dims[layer] for layer in layer_names]
        part_dim = dims[-1]

        for layer_name, dim in zip(layer_names, dims):
            layer = self._find_layer(layer_name)
            layer.register_forward_hook(self._hook)
            self.layers_fc.append(nn.Linear(dim, layer_proj_dim))

        self.fc = nn.Sequential(
            nn.Linear(sum(dims) + part_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    def _hook(self, _, __, output):
        self.layers_emb.append(output)

    def _clear_layers_emb(self):
        self.layers_emb = []

    def _find_layer(self, layer_name):
        modules = dict([*self.encoder.named_modules()])
        return modules.get(layer_name, None)

    def _get_part_feature(self, feat, part_proto):
        N, _, H, W = feat.shape
        M = part_proto.size(0)
        feat_flat = feat.flatten(2).permute((0, 2, 1))
        feat_norm = F.normalize(feat_flat, dim=-1)
        part_proto_norm = F.normalize(part_proto, dim=-1)
        feat_sim = (feat_norm @ part_proto_norm.T).permute((0, 2, 1))
        feat_sim = feat_sim.reshape((N, M, H, W))
        feat_parts = feat_sim.unsqueeze(2) * feat.unsqueeze(1)
        feat_parts = feat_parts.flatten(3).sum(-1)
        feat_part = feat_parts.mean(dim=1)
        return feat_part

    def forward(self, im, part_proto):
        if len(im.shape) == 5: # 训练模式 (SPA 增强)
            feats_proj = []
            for i in range(self.num_layers):
                _ = self.encoder(im[:, i])
                feat_layer_pool = self.pool(self.layers_emb[i])
                feats_proj.append(self.layers_fc[i](feat_layer_pool))
                self._clear_layers_emb()

            _ = self.encoder(im[:, self.num_layers])
            feat_spatial = self.layers_emb[-1] # 获取最后的特征图用于 CRA
            feat_layers_pool = [self.pool(each) for each in self.layers_emb]
            feat_global = torch.concatenate(feat_layers_pool, dim=1)
            feat_part = self._get_part_feature(feat_spatial, part_proto)
            feat_global_part = torch.concatenate([feat_global, feat_part], dim=1)
            feat_global_part_proj = self.fc(feat_global_part)
            self._clear_layers_emb()
            return feats_proj, feat_global_part_proj, feat_spatial
        else: # 评估模式
            _ = self.encoder(im)
            feat_spatial = self.layers_emb[-1]
            feat_global = self.pool(feat_spatial)
            feat_part = self._get_part_feature(feat_spatial, part_proto)
            feat_global_part = torch.concatenate([feat_global, feat_part], dim=1)
            return feat_global_part, feat_global

# ==========================================
# 3. 最终融合 CRA 的 PAPN 模型
# ==========================================
class PAPN_CRA(nn.Module):
    def __init__(self, base_encoder, proj_dim=256, layer_proj_dim=128, layer_names=None, 
                 K=4096, m=0.999, n_parts=5, T=0.15, pretrain=None, cra_alpha=1.0):
        super(PAPN_CRA, self).__init__()

        if layer_names is None:
            layer_names = ['layer2', 'layer3', 'layer4']

        self.K = K
        self.m = m
        self.T = T
        self.cra_alpha = cra_alpha

        self.encoder_q = Encoder(base_encoder, proj_dim, layer_proj_dim, layer_names, pretrain)
        self.encoder_k = Encoder(base_encoder, proj_dim, layer_proj_dim, layer_names, pretrain)

        # CRA 模块初始化 (针对 ResNet50 layer4)
        self.cra_module = MCRegionLoss(num_classes=196, feat_dim=2048)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(proj_dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.ssl_loss = nn.CrossEntropyLoss()
        self.sd_loss = AlignLoss() # 假设已有原 AlignLoss 定义

        self.part_proto = nn.Parameter(generate_orthonormal_vectors(n_parts, layer_dims[layer_names[-1]]))

    def forward(self, im_q, im_k):
        # 1. 前向传播获取投影特征和特征图
        q_feats, q, spatial_q = self.encoder_q(im_q, self.part_proto)
        q_norm = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k_feats, k, _ = self.encoder_k(im_k, self.part_proto)
            k_norm = F.normalize(k, dim=1)

            for i in range(len(k_feats)):
                k_feats[i] = self._batch_unshuffle_ddp(k_feats[i], idx_unshuffle)
            k_norm = self._batch_unshuffle_ddp(k_norm, idx_unshuffle)

        # 2. 计算 CRA Loss (核心 Trick 融入)
        cra_loss = self.cra_module(spatial_q)

        # 3. 计算 PAPN 原有 Loss
        sd_loss = self.sd_loss(q_feats, k_feats)
        l_pos = torch.einsum("nc,nc->n", [q_norm, k_norm]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q_norm, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        ssl_loss = self.ssl_loss(logits, labels)

        self._dequeue_and_enqueue(k_norm)

        # 融合后的总损失
        return ssl_loss + sd_loss + self.cra_alpha * cra_loss

    # 保持原有的辅助函数 (_momentum_update_key_encoder, _dequeue_and_enqueue 等)
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # 假设已导入 concat_all_gather
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        torch.distributed.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this]

# ==========================================
# 4. 其他辅助函数 (AlignLoss, concat_all_gather 等)
# ==========================================
class AlignLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()
    def forward(self, q_feats, k_feats):
        loss = 0
        for q, k in zip(q_feats, k_feats):
            loss += self.loss_fn(F.normalize(q, dim=-1), F.normalize(k, dim=-1))
        return loss

def generate_orthonormal_vectors(n, dim):
    A = torch.randn(dim, n)
    U, S, Vt = torch.svd(A)
    return U.T

def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)