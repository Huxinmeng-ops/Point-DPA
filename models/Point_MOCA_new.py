"""
Point-MOCA: 将 MOCA 自监督学习方法应用到点云
结合 Point-MAE 的点云处理架构和 MOCA 的 Masked Cross-view Completion 策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
import random
import copy
import math

from .build import MODELS
from utils import misc
from utils.logger import *
from knn_cuda import KNN

# ============================================================================
# 从 Point_MAE.py 导入的基础模块
# ============================================================================

class Encoder(nn.Module):  ## Embedding module
    
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG encoder_channel n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG encoder_channel
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):
    """FPS + KNN 点云分组 - 从 Point_MAE 复用，支持共享中心点"""
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, center=None):
        '''
            input: 
                xyz: B N 3
                center: B G 3 (可选，用于共享FPS采样)
            ---------------------------
            output: 
                neighborhood: B G M 3
                center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        
        # ✅ 如果提供了 center，使用它；否则使用 FPS 采样
        if center is None:
            center = misc.fps(xyz, self.num_group)  # B G 3
        
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder - 从 Point_MAE 复用"""
    def __init__(self, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class MaskTransformer(nn.Module):
    """
    Mask Transformer Encoder - Point_MAE 风格
    用于 Point_MOCA 的 student 和 teacher 编码器
    """
    def __init__(self, encoder_channel=384, trans_dim=384, depth=12, drop_path_rate=0.1, 
                 num_heads=6, mask_ratio=0.6, mask_type='rand'):
        super().__init__()
        self.trans_dim = trans_dim
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        
        # Patch encoder
        self.encoder = Encoder(encoder_channel=encoder_channel)
        
        # Position embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        
        self.norm = nn.LayerNorm(self.trans_dim)
    
    def _mask_center_rand(self, center, noaug=False):
        """随机 mask 策略"""
        B, G, _ = center.shape
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        
        num_mask = int(self.mask_ratio * G)
        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - num_mask),
                np.ones(num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        return overall_mask.to(center.device)
    
    def _mask_center_block(self, center, noaug=False):
        """块状 mask 策略"""
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        
        mask_idx = []
        for points in center:
            points = points.unsqueeze(0)
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1)
            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]
            mask_num = int(self.mask_ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())
        
        bool_masked_pos = torch.stack(mask_idx).to(center.device)
        return bool_masked_pos
    
    def forward(self, neighborhood, center, mask=None, noaug=False):
        """
        前向传播
        
        Args:
            neighborhood: B G M 3 - 分组后的点云
            center: B G 3 - 分组中心
            mask: B G (bool) - 可选的预定义 mask
            noaug: bool - 是否禁用 mask
        
        Returns:
            x: 编码后的特征 (B N C)
            mask: B G (bool) - 使用的 mask
            center_vis: B N_vis 3 - 可见 patch 的中心（如果使用了 mask）
        """
        # Encode patches
        group_input_tokens = self.encoder(neighborhood)  # B G C
        B, G, C = group_input_tokens.shape
        
        # Generate or use provided mask
        if mask is None and not noaug:
            if self.mask_type == 'rand':
                mask = self._mask_center_rand(center, noaug=noaug)
            else:
                mask = self._mask_center_block(center, noaug=noaug)
        elif mask is None:
            mask = torch.zeros(B, G).bool().to(center.device)
        
        # Select visible patches
        if mask.any():
            x_vis = group_input_tokens[~mask].reshape(B, -1, C)
            center_vis = center[~mask].reshape(B, -1, 3)
        else:
            x_vis = group_input_tokens
            center_vis = center
        
        # Position embedding
        pos = self.pos_embed(center_vis)
        x = x_vis
        
        # Transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        
        return x, mask, center_vis


# ============================================================================
# 从 MOCA 迁移的组件 - 用于在线 Codebook 和 Assignment
# ============================================================================

NORMALIZE_EPS = 1e-5


class L2Normalize(nn.Module):
    def __init__(self, dim):
        super(L2Normalize, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=NORMALIZE_EPS)


class BoWExtractor(nn.Module):
    """在线 Codebook 和 Bag-of-Words 提取器"""
    def __init__(
        self,
        num_words,
        num_channels,
        inv_delta=15,
        num_new_words=16,
        skip_offset=0,  # 点云中不需要跳过边界 patch
        update_type="random_token"):
        super(BoWExtractor, self).__init__()

        assert isinstance(inv_delta, (float, int))
        self.inv_delta = inv_delta
        self.Knew = num_new_words
        self.skip_offset = skip_offset
        self.decay = 0.99
        self.update_type = update_type
        assert update_type in ("random_token", "avg_token")

        # 初始化 embedding（codebook/visual words）
        # ✅ 修复：不使用 clamp(min=0)，保持正负值以匹配 normalized 特征
        embedding = torch.randn(num_words, num_channels)
        embedding = F.normalize(embedding, p=2, dim=1, eps=NORMALIZE_EPS)
        self.register_buffer("_embedding", embedding)
        self.register_buffer("_embedding_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_track_num_batches", torch.zeros(1))
        # ✅ 修复：使用更合理的初始值（基于理论值 sqrt(dim)）
        initial_dist_norm = math.sqrt(num_channels) * 0.5
        self.register_buffer("_dist_norm", torch.ones(1) * initial_dist_norm)
        self.register_buffer("_dist_norm_prev", torch.ones(1) * initial_dist_norm)
        self._dist_norm_prev.data.copy_(self._dist_norm.data)

    @torch.no_grad()
    def update(self, x, attn=None):
        """更新在线 codebook"""
        assert self.training
        N, L, C = x.size()  # N: batch_size, L: num_patches, C: channels
        Knew = self.Knew // max(1, misc.get_world_size() if hasattr(misc, 'get_world_size') else 1)
        Knew = min(Knew, N)
        
        # 随机选择 Knew 个样本
        batch_idx = torch.randperm(N)[:Knew].long().to(x.device)

        if self.update_type == "random_token":
            # 随机选择每个样本的一个 patch token
            # ✅ 修复：使用 reshape 而不是 view，因为 x 可能不连续
            x = x.reshape(N * L, C)
            token_idx = torch.randint(0, L, (Knew,), device=x.device)
            new_words = x[batch_idx * L + token_idx]  # [Knew, C]
        elif self.update_type == "avg_token":
            # 使用平均 token
            new_words = x[batch_idx].mean(dim=1)  # [Knew, C]

        new_words = F.normalize(new_words, p=2, dim=1, eps=NORMALIZE_EPS)

        # 更新 codebook
        assert self._embedding.shape[0] % new_words.shape[0] == 0
        ptr = int(self._embedding_ptr)
        self._embedding[ptr:(ptr + new_words.shape[0]), :] = new_words
        self._embedding_ptr[0] = (ptr + new_words.shape[0]) % self._embedding.shape[0]

        self._dist_norm_prev.data.copy_(self._dist_norm.data)
        self._track_num_batches += 1

    @torch.no_grad()
    def get_dictionary(self):
        """返回当前的 codebook"""
        return self._embedding.detach().clone()

    def compute_bow(self, codes):
        """计算 Bag-of-Words 向量（全局 pooling）"""
        # codes shape: [N, L, K]
        bow = codes.mean(dim=1)  # [N, K]
        bow = F.normalize(bow, p=1, dim=1, eps=NORMALIZE_EPS)  # L1-normalization
        return bow

    def assign_words(self, x):
        """将特征分配到最近的 visual word"""
        # x shape: [N, L, C]
        words = self._embedding  # [K, C]
        x = F.normalize(x, p=2, dim=2, eps=NORMALIZE_EPS)
        dist = -torch.nn.functional.linear(x, weight=words, bias=None)

        dist = dist.float()
        # dist shape: [N, L, K]
        min_dist, enc_indices = torch.min(dist, dim=2)  # [N, L]
        
        if self.training:
            # EMA 更新距离归一化因子
            self._ave_min_dist = min_dist.mean().item()
            dist_norm_tmp = (torch.mean(dist, dim=2) - min_dist).mean()
            dist_norm_tmp = dist_norm_tmp.abs()
            self._dist_norm.data.mul_(self.decay).add_(dist_norm_tmp, alpha=(1. - self.decay))

        # Soft assignment codes
        # ✅ 优化：添加数值稳定性保护，同时限制上限
        dist_norm_safe = torch.clamp(self._dist_norm_prev, min=0.1, max=10.0)  # 同时限制上限
        inv_delta = self.inv_delta / dist_norm_safe
        # ✅ 修复：降低上限，避免分布过于尖锐导致损失过小
        # 当 inv_delta 过大时，softmax 会变得过于尖锐（接近 one-hot），
        # 导致预测和目标分布都接近 one-hot 时，KL divergence 变得非常小
        inv_delta = torch.clamp(inv_delta, min=0.1, max=10.0)  # 从 50.0 降低到 10.0
        codes = F.softmax(-inv_delta * dist, dim=2)
        return codes

    def forward(self, x):
        """
        Input: x: [N, L, C]
        Output: bow: [N, K], codes: [N, L, K]
        """
        codes = self.assign_words(x)
        bow = self.compute_bow(codes)
        return bow, codes


class BoWExtractorMultipleLevels(nn.Module):
    """多层次 BoW 提取器"""
    def __init__(self, opts_list, bow_fn=BoWExtractor):
        super(BoWExtractorMultipleLevels, self).__init__()
        assert isinstance(opts_list, (list, tuple))
        self.bow_extractor = nn.ModuleList([bow_fn(**opts) for opts in opts_list])

    @torch.no_grad()
    def get_dictionary(self):
        return [b.get_dictionary() for b in self.bow_extractor]

    def forward(self, features):
        assert isinstance(features, (list, tuple))
        assert len(features) == len(self.bow_extractor)
        out = list(zip(*[b(f) for b, f in zip(self.bow_extractor, features)]))
        return out

    def update(self, features, attn=None):
        assert isinstance(features, (list, tuple))
        assert len(features) == len(self.bow_extractor)
        for b, f in zip(self.bow_extractor, features):
            b.update(f, attn)


class ResWGEN(nn.Module):
    """残差权重生成器"""
    def __init__(self, generator, num_channels_in, num_channels_out):
        super(ResWGEN, self).__init__()
        self.l2norm = L2Normalize(dim=1)
        self.generator = generator
        if num_channels_in == num_channels_out:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(num_channels_in, num_channels_out)

    def forward(self, dictionary):
        x = self.l2norm(dictionary)
        x_res = self.generator(x)
        x_skip = self.skip(x)
        return self.l2norm(x_res + x_skip)


class BoWPredictor(nn.Module):
    """动态 BoW 预测头"""
    def __init__(
        self,
        num_channels_out=384,
        num_channels_in=[384,],
        num_channels_hidden=1024,
        kappa=8,
        learn_kappa=False,
        num_layers=2,
        residual=True,
    ):
        super(BoWPredictor, self).__init__()
        assert num_layers >= 1
        assert isinstance(num_channels_in, (list, tuple))
        num_code_levels = len(num_channels_in)
        assert num_code_levels == 1

        bottleneck_dim = num_channels_out

        generators = nn.Sequential()
        if residual is False:
            generators.add_module(f"b0_l2norm_in", L2Normalize(dim=1))
        if num_layers == 1:
            num_channels_last = num_channels_in[0]
        else:
            generators.add_module(f"b0_fc", nn.Linear(num_channels_in[0], num_channels_hidden, bias=False))
            generators.add_module(f"b0_bn", nn.BatchNorm1d(num_channels_hidden))
            generators.add_module(f"b0_rl", nn.ReLU(inplace=False))
            for layer in range(2, num_layers):
                generators.add_module(f"b0_fc{layer}", nn.Linear(num_channels_hidden, num_channels_hidden, bias=False))
                generators.add_module(f"b0_bn{layer}", nn.BatchNorm1d(num_channels_hidden))
                generators.add_module(f"b0_rl{layer}", nn.ReLU(inplace=False))

            num_channels_last = num_channels_hidden
        generators.add_module(f"b0_last_layer", nn.Linear(num_channels_last, bottleneck_dim))
        if residual is False:
            generators.add_module(f"b0_l2norm_out", L2Normalize(dim=1))
        else:
            generators = ResWGEN(generators, num_channels_in[0], bottleneck_dim)

        self.layers_w = nn.ModuleList([generators,])

        self.scale = nn.Parameter(
            torch.FloatTensor(num_code_levels).fill_(kappa),
            requires_grad=learn_kappa)

    def forward(self, features, dictionary):
        """动态预测 BoW"""
        assert isinstance(dictionary, (list, tuple))
        assert len(dictionary) == len(self.layers_w)

        weight = [gen(dict).t() for gen, dict in zip(self.layers_w, dictionary)]
        kappa = torch.split(self.scale, 1, dim=0)

        if isinstance(features, torch.Tensor):
            preds = [torch.mm(features * k, w) for k, w in zip(kappa, weight)]
        else:
            preds = [[torch.mm(f * k, w) for k, w in zip(kappa, weight)] for f in features]

        return preds


class TransformerDecoder(nn.Module):
    """Transformer Decoder - 用于局部 patch assignment 预测"""
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        x = self.norm(x)
        return x


# ============================================================================
# Point-MOCA 主模型
# ============================================================================

@MODELS.register_module()
class Point_MOCA_old(nn.Module):
    """
    Point-MOCA: 将 MOCA 的 Masked Cross-view Completion 应用到点云
    
    核心思路：
    1. 使用 Point-MAE 的 tokenization（FPS + KNN 分组）
    2. Teacher：从完整点云生成 patch assignment（基于在线 codebook）
    3. Student：从 masked 点云预测 patch assignment
    4. 损失：局部 patch assignment + 全局 BoW 一致性
    
    ✅ 特征提取：直接使用 encoder + blocks
    """
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MOCA] Building Point-MOCA model', logger='Point_MOCA')
        self.config = config
        
        # ====== Point-MAE 风格的配置 ======
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        self.encoder_dims = config.transformer_config.encoder_dims
        
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.mask_ratio = config.transformer_config.mask_ratio
        self.mask_type = config.transformer_config.mask_type
        
        # ====== MOCA 风格的配置 ======
        self.inv_delta = config.get('inv_delta', 10.0)
        self.num_words = config.get('num_words', 4096)
        self.num_new_words = config.get('num_new_words', 16)
        self.kappa = config.get('kappa', 5.0)
        self.pred_mlp_ratio = config.get('pred_mlp_ratio', 2)
        self.use_loc_loss = config.get('use_loc_loss', True)
        
        # ====== 点云分组（Tokenization）======
        print_log(f'[Point_MOCA] Divide point cloud into G{self.num_group} x S{self.group_size} points', 
                  logger='Point_MOCA')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        
        # ====== Student Encoder (直接定义组件) ======
        print_log(f'[Point_MOCA] Building Student encoder components', logger='Point_MOCA')
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        
        # Position embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        
        # ✅ 优化：添加全局特征投影层（用于 max+mean pooling）
        self.global_proj = nn.Sequential(
            nn.Linear(self.trans_dim * 2, self.trans_dim),
            nn.LayerNorm(self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.trans_dim)
        )
        
        # ====== Teacher Encoder (EMA，直接定义组件) ======
        print_log(f'[Point_MOCA] Building Teacher encoder components (EMA)', logger='Point_MOCA')
        self.encoder_teacher = Encoder(encoder_channel=self.encoder_dims)
        self.pos_embed_teacher = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        self.blocks_teacher = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.norm_teacher = nn.LayerNorm(self.trans_dim)
        
        # ✅ 优化：Teacher 也需要全局特征投影层（用于一致性）
        self.global_proj_teacher = nn.Sequential(
            nn.Linear(self.trans_dim * 2, self.trans_dim),
            nn.LayerNorm(self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.trans_dim)
        )
        # 初始化 teacher 的投影层
        for param_s, param_t in zip(self.global_proj.parameters(), self.global_proj_teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
        
        # ====== 在线 Codebook 和 BoW Extractor ======
        code_extractor_opts_list = [{
            "num_channels": self.trans_dim,
            "inv_delta": self.inv_delta,
            "num_words": self.num_words,
            "num_new_words": self.num_new_words,
            "skip_offset": 0,  # 点云不需要跳过边界
            "update_type": "random_token"
        }]
        self.bow_extractor = BoWExtractorMultipleLevels(code_extractor_opts_list)
        
        # ====== Student Predictor（预测全局 BoW）======
        code_predictor_opts = {
            "kappa": self.kappa,
            "num_channels_out": self.trans_dim,
            "num_channels_hidden": int(self.trans_dim * self.pred_mlp_ratio),
            "num_channels_in": [self.trans_dim,],
            "residual": True
        }
        self.encoder_pred = BoWPredictor(**code_predictor_opts)
        
        # ====== Decoder（用于局部 patch assignment 预测）======
        if self.use_loc_loss:
            self.decoder_depth = config.transformer_config.get('decoder_depth', 4)
            self.decoder_num_heads = config.transformer_config.get('decoder_num_heads', 6)
            
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.decoder_pos_embed = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, self.trans_dim)
            )
            
            dpr_dec = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
            self.decoder_blocks = TransformerDecoder(
                embed_dim=self.trans_dim,
                depth=self.decoder_depth,
                drop_path_rate=dpr_dec,
                num_heads=self.decoder_num_heads,
            )
            
            code_predictor_opts_decoder = copy.deepcopy(code_predictor_opts)
            code_predictor_opts_decoder["num_channels_in"] = [self.trans_dim,]
            code_predictor_opts_decoder["num_channels_out"] = self.trans_dim
            self.decoder_pred = BoWPredictor(**code_predictor_opts_decoder)
            
            trunc_normal_(self.mask_token, std=.02)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 初始化 teacher 为 student 的副本（EMA）
        print_log(f'[Point_MOCA] Initializing Teacher as copy of Student', logger='Point_MOCA')
        for param_s, param_t in zip(self.encoder.parameters(), self.encoder_teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
        for param_s, param_t in zip(self.pos_embed.parameters(), self.pos_embed_teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
        for param_s, param_t in zip(self.blocks.parameters(), self.blocks_teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
        for param_s, param_t in zip(self.norm.parameters(), self.norm_teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
        
        print_log(f'[Point_MOCA] Model initialized successfully', logger='Point_MOCA')
        print_log(f'[Point_MOCA] use_loc_loss={self.use_loc_loss}', logger='Point_MOCA')
        print_log(f'[Point_MOCA] Feature extraction: max+mean pooling with projection', logger='Point_MOCA')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def update_teacher(self, momentum):
        """EMA 更新 teacher 网络"""
        if not self.training:
            return
        if momentum >= 1.0:
            return
        
        # 更新 encoder
        for param_s, param_t in zip(self.encoder.parameters(), self.encoder_teacher.parameters()):
            if param_s.requires_grad:
                param_t.data = param_t.data * momentum + param_s.detach().data * (1. - momentum)
        
        # 更新 pos_embed
        for param_s, param_t in zip(self.pos_embed.parameters(), self.pos_embed_teacher.parameters()):
            if param_s.requires_grad:
                param_t.data = param_t.data * momentum + param_s.detach().data * (1. - momentum)
        
        # 更新 blocks
        for param_s, param_t in zip(self.blocks.parameters(), self.blocks_teacher.parameters()):
            if param_s.requires_grad:
                param_t.data = param_t.data * momentum + param_s.detach().data * (1. - momentum)
        
        # 更新 norm
        for param_s, param_t in zip(self.norm.parameters(), self.norm_teacher.parameters()):
            if param_s.requires_grad:
                param_t.data = param_t.data * momentum + param_s.detach().data * (1. - momentum)
        
        # ✅ 优化：更新全局特征投影层
        for param_s, param_t in zip(self.global_proj.parameters(), self.global_proj_teacher.parameters()):
            if param_s.requires_grad:
                param_t.data = param_t.data * momentum + param_s.detach().data * (1. - momentum)

    def forward_encoder_student(self, neighborhood, center, mask):
        """
        Student encoder: 对可见的 patches 进行编码
        """
        B = neighborhood.shape[0]
        
        # 先编码 patches
        group_tokens = self.encoder(neighborhood)  # B G C
        
        # Early Masking: 在编码后立即 mask
        x_vis = group_tokens[~mask].reshape(B, -1, self.trans_dim)  # B N_vis C
        
        # 获取可见 patches 的中心和位置编码
        center_vis = center[~mask].reshape(B, -1, 3)  # B N_vis 3
        pos_vis = self.pos_embed(center_vis)  # B N_vis C
        
        # Transformer blocks
        x = self.blocks(x_vis, pos_vis)  # B N_vis C
        x = self.norm(x)  # B N_vis C
        
        # ✅ 优化：使用 max + mean pooling 的拼接，保留更多信息
        x_mean = x.mean(dim=1)  # B C
        x_max = x.max(dim=1)[0]  # B C
        x_global = torch.cat([x_mean, x_max], dim=-1)  # B 2C
        # 如果维度不匹配，使用投影层（在 __init__ 中添加）
        if hasattr(self, 'global_proj'):
            x_global = self.global_proj(x_global)  # B C
        
        return x, x_global, center_vis

    @torch.no_grad()
    def forward_encoder_teacher(self, neighborhood, center):
        """
        Teacher encoder: 对完整点云进行编码
        """
        # 先编码 patches
        group_tokens = self.encoder_teacher(neighborhood)  # B G C
        
        # Teacher 不使用 mask，直接处理所有 patches
        pos = self.pos_embed_teacher(center)  # B G C
        
        # Transformer blocks
        x = self.blocks_teacher(group_tokens, pos)  # B G C
        x = self.norm_teacher(x)  # B G C
        
        # ✅ 优化：Teacher 也使用 max+mean pooling（与 student 一致）
        x_mean = x.mean(dim=1)  # B C
        x_max = x.max(dim=1)[0]  # B C
        x_global = torch.cat([x_mean, x_max], dim=-1)  # B 2C
        x_global = self.global_proj_teacher(x_global)  # B C
        
        # 返回 patch-level 特征（用于 codebook assignment）和全局特征
        # 注意：这里返回的是 patch tokens，不是全局特征
        return x.contiguous()

    def forward_decoder(self, x_vis, center_vis, center, mask):
        """Decoder: 预测被 mask 的 patches 的 assignment"""
        B, _, C = x_vis.shape
        
        # x_vis 已经是 patch tokens，直接使用
        x_vis_patches = x_vis
        
        # 准备 mask tokens
        center_mask = center[mask].reshape(B, -1, 3)
        num_mask = center_mask.shape[1]
        mask_tokens = self.mask_token.expand(B, num_mask, -1)
        
        # 拼接 visible 和 mask tokens
        x_full = torch.cat([x_vis_patches, mask_tokens], dim=1)
        
        # Position embeddings
        pos_vis = self.decoder_pos_embed(center_vis)
        pos_mask = self.decoder_pos_embed(center_mask)
        pos_full = torch.cat([pos_vis, pos_mask], dim=1)
        
        # Decoder transformer
        x_decoded = self.decoder_blocks(x_full, pos_full)
        
        # 只返回 mask 部分的特征
        x_mask_decoded = x_decoded[:, -num_mask:, :]  # B num_mask C
        
        return x_mask_decoded, center_mask

    def extract_targets(self, pts1, pts2, momentum, update_teacher=True):
        """
        使用 teacher 提取目标 assignments
        参考 moca.py 的实现：teacher 更新在 extract_targets 内部进行
        """
        with torch.no_grad():
            if update_teacher:
                self.update_teacher(momentum)  # ✅ 修复：在 extract_targets 内部更新 teacher（与 moca.py 一致）
            
            # 获取当前 codebook
            dictionary = self.bow_extractor.get_dictionary()
            
            # Teacher 编码两个视图（使用直接的特征提取方式：encoder + blocks）
            neighborhood1, center1 = self.group_divider(pts1)
            neighborhood2, center2 = self.group_divider(pts2)
            
            teacher_features1 = self.forward_encoder_teacher(neighborhood1, center1)
            teacher_features2 = self.forward_encoder_teacher(neighborhood2, center2)
            
            # 提取 BoW 和 codes
            bow_code_x1, codes_x1 = self.bow_extractor([teacher_features1])
            bow_code_x2, codes_x2 = self.bow_extractor([teacher_features2])
        
        # ✅ 修复：codebook 更新策略与 moca.py 一致（每个 batch 都更新，50% 概率随机选择视图）
        if self.training and update_teacher:
            # Update the teacher's codebook / dictionary (与 moca.py 一致)
            if random.random() < 0.5:
                self.bow_extractor.update([teacher_features1])
            else:
                self.bow_extractor.update([teacher_features2])
        
        # Cross-view targets: view1 预测 view2 的 BoW，反之亦然
        same_view_codes = torch.cat([codes_x1[0], codes_x2[0]], dim=0)
        cross_view_bows = torch.cat([bow_code_x2[0], bow_code_x1[0]], dim=0)
        
        return cross_view_bows, same_view_codes, dictionary

    def forward_img_loss(self, target, pred):
        """全局 BoW 预测损失（Cross-view）"""
        # ✅ 修复：添加温度缩放，防止分布过于尖锐导致损失过小
        # target: (B, K) - soft assignment probabilities
        # pred: (B, K) - logits
        
        # 添加温度缩放，使预测分布更平滑（防止过于尖锐）
        temperature = 1.0  # 可以调整，1.0 表示不缩放
        pred_scaled = pred / temperature
        pred_log_softmax = F.log_softmax(pred_scaled, dim=1)
        
        # 添加数值稳定性：避免 target 中有 0 导致 log(0)
        target_smooth = target + 1e-8
        target_smooth = target_smooth / target_smooth.sum(dim=1, keepdim=True)
        
        # ✅ 修复：添加最小损失阈值，防止损失过小
        loss = F.kl_div(pred_log_softmax, target_smooth, reduction="batchmean", log_target=False)
        
        # ✅ 修复：如果损失过小，可能是分布过于尖锐，添加熵正则项
        # 计算预测分布的熵，如果熵过小（分布过于尖锐），增加惩罚
        pred_probs = F.softmax(pred_scaled, dim=1)
        entropy = -(pred_probs * pred_log_softmax).sum(dim=1).mean()
        min_entropy = 1.0  # 最小熵阈值（对于 K=4096，均匀分布的熵约为 log(4096) ≈ 8.3）
        if entropy < min_entropy:
            # 分布过于尖锐，添加熵正则项（增加系数，使惩罚更明显）
            entropy_penalty = 0.1 * (min_entropy - entropy)
            loss = loss + entropy_penalty
        
        # ✅ 修复：如果熵过小且损失过小，可能是分布过于尖锐，增加最小损失
        # 注意：只在熵过小时才添加最小损失，避免影响正常学习
        min_loss = torch.tensor(1e-4, device=loss.device, dtype=loss.dtype)
        if entropy < min_entropy:
            # 分布过于尖锐时，确保损失不会过小
            loss = torch.clamp(loss, min=min_loss)
        
        return loss

    def forward_loc_loss(self, target, pred):
        """局部 patch assignment 损失（Same-view）"""
        # target: (N, K) or (B, G, K)
        # pred: (N, K)
        if target.dim() == 3:
            # 如果 target 是 3D，展平它
            K = target.shape[2]
            target = target.view(-1, K)
        
        # ✅ 修复：添加温度缩放，防止分布过于尖锐导致损失过小
        temperature = 1.0  # 可以调整，1.0 表示不缩放
        pred_scaled = pred / temperature
        pred_log_softmax = F.log_softmax(pred_scaled, dim=1)
        
        # 添加数值稳定性
        target_smooth = target + 1e-8
        target_smooth = target_smooth / target_smooth.sum(dim=1, keepdim=True)
        loss = F.kl_div(pred_log_softmax, target_smooth, reduction="batchmean", log_target=False)
        
        # ✅ 修复：如果损失过小，可能是分布过于尖锐，添加熵正则项
        # 计算预测分布的熵，如果熵过小（分布过于尖锐），增加惩罚
        pred_probs = F.softmax(pred_scaled, dim=1)
        entropy = -(pred_probs * pred_log_softmax).sum(dim=1).mean()
        min_entropy = 1.0  # 最小熵阈值
        if entropy < min_entropy:
            # 分布过于尖锐，添加熵正则项（增加系数，使惩罚更明显）
            entropy_penalty = 0.1 * (min_entropy - entropy)
            loss = loss + entropy_penalty
        
        # ✅ 修复：如果熵过小且损失过小，可能是分布过于尖锐，增加最小损失
        # 注意：只在熵过小时才添加最小损失，避免影响正常学习
        min_loss = torch.tensor(1e-4, device=loss.device, dtype=loss.dtype)
        if entropy < min_entropy:
            # 分布过于尖锐时，确保损失不会过小
            loss = torch.clamp(loss, min=min_loss)
        
        return loss

    def _mask_center_rand(self, center, noaug=False):
        """随机 mask 策略"""
        B, G, _ = center.shape
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool().to(center.device)
        
        num_mask = int(self.mask_ratio * G)
        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - num_mask),
                np.ones(num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        return overall_mask.to(center.device)

    def _mask_center_block(self, center, noaug=False):
        """块状 mask 策略"""
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool().to(center.device)
        
        mask_idx = []
        for points in center:
            points = points.unsqueeze(0)
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1)
            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]
            mask_num = int(self.mask_ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())
        
        bool_masked_pos = torch.stack(mask_idx).to(center.device)
        return bool_masked_pos

    def forward(self, pts, vis=False, **kwargs):
        """
        前向传播
        
        训练模式：
            pts: 点云数据 [B, N, 3]
            返回 loss
        
        可视化模式：
            vis=True
            返回可视化数据
        """
        if not self.training or vis:
            # 推理或可视化模式
            neighborhood, center = self.group_divider(pts)
            mask = self._mask_center_rand(center) if self.mask_type == 'rand' else self._mask_center_block(center)
            
            # 使用直接的特征提取方式：encoder + blocks
            x_vis, x_global, center_vis = self.forward_encoder_student(neighborhood, center, mask)
            
            if vis:
                # 返回可视化数据（类似 Point-MAE）
                B = pts.shape[0]
                M = mask.sum(dim=1)[0].item()
                vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                full_vis = vis_points + center[~mask].reshape(-1, 1, 3)
                return full_vis.reshape(B, -1, 3), center
            else:
                return x_global
        
        # 训练模式
        momentum = kwargs.get('momentum', 0.996)
        img_weight = kwargs.get('img_weight', 1.0)
        loc_weight = kwargs.get('loc_weight', 1.0)
        
        # ✅ 修复：生成两个不同的 masked views（使用不同的 mask）
        # View 1: 使用一个 mask
        neighborhood, center = self.group_divider(pts)
        mask1 = self._mask_center_rand(center) if self.mask_type == 'rand' else self._mask_center_block(center)
        
        # View 2: 使用另一个不同的 mask（确保不同）
        mask2 = self._mask_center_rand(center) if self.mask_type == 'rand' else self._mask_center_block(center)
        # 确保两个 mask 不完全相同（至少有一些差异）
        max_attempts = 10
        attempt = 0
        while attempt < max_attempts and torch.allclose(mask1.float(), mask2.float()):
            mask2 = self._mask_center_rand(center) if self.mask_type == 'rand' else self._mask_center_block(center)
            attempt += 1
        
        # ✅ 修复：提取 teacher targets（与 moca.py 一致，teacher 更新在 extract_targets 内部进行）
        # 使用相同的点云但不同的 mask 来创建 cross-view
        cross_view_bows, same_view_codes, dictionary = self.extract_targets(pts, pts, momentum, update_teacher=True)
        
        # Student 编码两个 masked views（使用直接的特征提取方式：encoder + blocks）
        x_vis1, x_global1, center_vis1 = self.forward_encoder_student(neighborhood, center, mask1)
        x_vis2, x_global2, center_vis2 = self.forward_encoder_student(neighborhood, center, mask2)
        
        # 拼接两个 views
        x_global_both = torch.cat([x_global1, x_global2], dim=0)
        
        # ====== 全局 BoW 预测损失（Cross-view）======
        bow_preds = self.encoder_pred(x_global_both, dictionary)
        loss_img = [self.forward_img_loss(cross_view_bows, pred) for pred in bow_preds]
        loss_img = torch.stack(loss_img).mean() * 2
        loss_tot = loss_img * img_weight
        
        # ====== 局部 patch assignment 损失（Same-view）======
        if self.use_loc_loss:
            # Decoder 预测 masked patches 的 assignment
            x_mask_decoded1, _ = self.forward_decoder(x_vis1, center_vis1, center, mask1)
            x_mask_decoded2, _ = self.forward_decoder(x_vis2, center_vis2, center, mask2)
            
            # 从 same_view_codes 中提取 masked patches 的 target codes
            # same_view_codes: (2B, G, K) - 包含两个视图的所有 patches
            B = pts.shape[0]
            codes_view1 = same_view_codes[:B]  # (B, G, K)
            codes_view2 = same_view_codes[B:]  # (B, G, K)
            
            # 使用 mask 选择被遮挡的 patches
            # mask1, mask2: (B, G) boolean
            # 需要将其重塑为 (B*G,) 然后选择
            target_codes1 = codes_view1.reshape(-1, codes_view1.shape[-1])[mask1.reshape(-1)]
            target_codes2 = codes_view2.reshape(-1, codes_view2.shape[-1])[mask2.reshape(-1)]
            
            # 拼接预测和目标
            x_mask_decoded = torch.cat([
                x_mask_decoded1.reshape(-1, self.trans_dim),
                x_mask_decoded2.reshape(-1, self.trans_dim)
            ], dim=0)
            
            target_codes_masked = torch.cat([target_codes1, target_codes2], dim=0)
            
            # 预测
            codes_preds = self.decoder_pred(x_mask_decoded, dictionary)
            
            # 计算损失
            loss_loc = [self.forward_loc_loss(target_codes_masked, pred) for pred in codes_preds]
            loss_loc = torch.stack(loss_loc).mean() * 2
        else:
            loss_loc = torch.zeros_like(loss_img)
        
        loss_tot += (loss_loc * loc_weight)
        
        # 统计信息
        if hasattr(self, '_iter_count'):
            self._iter_count += 1
        else:
            self._iter_count = 0
        
        if self._iter_count % 100 == 0:
            print_log(f'[Point_MOCA] Iter {self._iter_count}: loss_img={loss_img.item():.4f}, '
                     f'loss_loc={loss_loc.item():.4f}, loss_tot={loss_tot.item():.4f}',
                     logger='Point_MOCA')
        
        return loss_tot

def augment_point_cloud(xyz):
        """
        对点云进行随机增强：缩放 + 旋转 + 抖动
        Input: xyz [B, N, 3]
        Output: augmented_xyz [B, N, 3]
        """
        B, N, _ = xyz.shape
        device = xyz.device
        
        # 1. 随机缩放 (Scaling): 范围 [0.8, 1.2]
        scales = torch.rand(B, 1, 1, device=device) * 0.4 + 0.8
        xyz = xyz * scales
        
        # 2. 随机旋转 (Rotation around Z-axis): 范围 [0, 2pi]
        # 即使是 ShapeNet 这种对齐的数据集，适当旋转也能增强鲁棒性
        thetas = torch.rand(B, device=device) * 2 * np.pi
        c, s = torch.cos(thetas), torch.sin(thetas)
        
        # 构建旋转矩阵 [B, 3, 3]
        # 假设 Z 轴向上，绕 Z 轴旋转
        zeros = torch.zeros_like(c)
        ones = torch.ones_like(c)
        rot_mats = torch.stack([
            c, -s, zeros,
            s,  c, zeros,
            zeros, zeros, ones
        ], dim=1).reshape(B, 3, 3)
        
        # 应用旋转: (B, N, 3) @ (B, 3, 3) -> (B, N, 3)
        xyz = torch.bmm(xyz, rot_mats)
        
        # 3. 随机抖动 (Jittering):添加微小噪声
        # 这是一个常见的 PointNet++ 增强策略
        noise = torch.randn(B, N, 3, device=device) * 0.01
        noise = torch.clamp(noise, -0.05, 0.05)
        xyz = xyz + noise
        
        return xyz
@MODELS.register_module()
class Point_MOCA_new(nn.Module):
    """
    Point-MOCA: 将 MOCA 的 Masked Cross-view Completion 应用到点云
    
    核心思路：
    1. 使用 Point-MAE 的 tokenization（FPS + KNN 分组）
    2. Teacher：从完整点云生成 patch assignment（基于在线 codebook）
    3. Student：从 masked 点云预测 patch assignment
    4. 损失：局部 patch assignment + 全局 BoW 一致性
    
    ✅ 特征提取：直接使用 encoder + blocks
    """
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MOCA] Building Point-MOCA model', logger='Point_MOCA')
        self.config = config
        
        # ====== Point-MAE 风格的配置 ======
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        self.encoder_dims = config.transformer_config.encoder_dims
        
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.mask_ratio = config.transformer_config.mask_ratio
        self.mask_type = config.transformer_config.mask_type
        
        # ====== MOCA 风格的配置 ======
        self.inv_delta = config.get('inv_delta', 10.0)
        self.num_words = config.get('num_words', 4096)
        self.num_new_words = config.get('num_new_words', 16)
        self.kappa = config.get('kappa', 5.0)
        self.pred_mlp_ratio = config.get('pred_mlp_ratio', 2)
        self.use_loc_loss = config.get('use_loc_loss', True)
        
        # ====== 点云分组（Tokenization）======
        print_log(f'[Point_MOCA] Divide point cloud into G{self.num_group} x S{self.group_size} points', 
                  logger='Point_MOCA')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        
        # ====== Student Encoder (直接定义组件) ======
        print_log(f'[Point_MOCA] Building Student encoder components', logger='Point_MOCA')
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        
        # Position embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        
        # ✅ 优化：添加全局特征投影层（用于 max+mean pooling）
        self.global_proj = nn.Sequential(
            nn.Linear(self.trans_dim * 2, self.trans_dim),
            nn.LayerNorm(self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.trans_dim)
        )
        
        # ====== Teacher Encoder (EMA，直接定义组件) ======
        print_log(f'[Point_MOCA] Building Teacher encoder components (EMA)', logger='Point_MOCA')
        self.encoder_teacher = Encoder(encoder_channel=self.encoder_dims)
        self.pos_embed_teacher = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        self.blocks_teacher = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.norm_teacher = nn.LayerNorm(self.trans_dim)
        
        # ✅ 优化：Teacher 也需要全局特征投影层（用于一致性）
        self.global_proj_teacher = nn.Sequential(
            nn.Linear(self.trans_dim * 2, self.trans_dim),
            nn.LayerNorm(self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.trans_dim)
        )
        # 初始化 teacher 的投影层
        for param_s, param_t in zip(self.global_proj.parameters(), self.global_proj_teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
        
        # ====== 在线 Codebook 和 BoW Extractor ======
        code_extractor_opts_list = [{
            "num_channels": self.trans_dim,
            "inv_delta": self.inv_delta,
            "num_words": self.num_words,
            "num_new_words": self.num_new_words,
            "skip_offset": 0,  # 点云不需要跳过边界
            "update_type": "random_token"
        }]
        self.bow_extractor = BoWExtractorMultipleLevels(code_extractor_opts_list)
        
        # ====== Student Predictor（预测全局 BoW）======
        code_predictor_opts = {
            "kappa": self.kappa,
            "num_channels_out": self.trans_dim,
            "num_channels_hidden": int(self.trans_dim * self.pred_mlp_ratio),
            "num_channels_in": [self.trans_dim,],
            "residual": True
        }
        self.encoder_pred = BoWPredictor(**code_predictor_opts)
        
        # ====== Decoder（用于局部 patch assignment 预测）======
        if self.use_loc_loss:
            self.decoder_depth = config.transformer_config.get('decoder_depth', 4)
            self.decoder_num_heads = config.transformer_config.get('decoder_num_heads', 6)
            
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.decoder_pos_embed = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, self.trans_dim)
            )
            
            dpr_dec = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
            self.decoder_blocks = TransformerDecoder(
                embed_dim=self.trans_dim,
                depth=self.decoder_depth,
                drop_path_rate=dpr_dec,
                num_heads=self.decoder_num_heads,
            )
            
            code_predictor_opts_decoder = copy.deepcopy(code_predictor_opts)
            code_predictor_opts_decoder["num_channels_in"] = [self.trans_dim,]
            code_predictor_opts_decoder["num_channels_out"] = self.trans_dim
            self.decoder_pred = BoWPredictor(**code_predictor_opts_decoder)
            
            trunc_normal_(self.mask_token, std=.02)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 初始化 teacher 为 student 的副本（EMA）
        print_log(f'[Point_MOCA] Initializing Teacher as copy of Student', logger='Point_MOCA')
        for param_s, param_t in zip(self.encoder.parameters(), self.encoder_teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
        for param_s, param_t in zip(self.pos_embed.parameters(), self.pos_embed_teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
        for param_s, param_t in zip(self.blocks.parameters(), self.blocks_teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
        for param_s, param_t in zip(self.norm.parameters(), self.norm_teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
        
        print_log(f'[Point_MOCA] Model initialized successfully', logger='Point_MOCA')
        print_log(f'[Point_MOCA] use_loc_loss={self.use_loc_loss}', logger='Point_MOCA')
        print_log(f'[Point_MOCA] Feature extraction: max+mean pooling with projection', logger='Point_MOCA')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def update_teacher(self, momentum):
        """EMA 更新 teacher 网络"""
        if not self.training:
            return
        if momentum >= 1.0:
            return
        
        # 更新 encoder
        for param_s, param_t in zip(self.encoder.parameters(), self.encoder_teacher.parameters()):
            if param_s.requires_grad:
                param_t.data = param_t.data * momentum + param_s.detach().data * (1. - momentum)
        
        # 更新 pos_embed
        for param_s, param_t in zip(self.pos_embed.parameters(), self.pos_embed_teacher.parameters()):
            if param_s.requires_grad:
                param_t.data = param_t.data * momentum + param_s.detach().data * (1. - momentum)
        
        # 更新 blocks
        for param_s, param_t in zip(self.blocks.parameters(), self.blocks_teacher.parameters()):
            if param_s.requires_grad:
                param_t.data = param_t.data * momentum + param_s.detach().data * (1. - momentum)
        
        # 更新 norm
        for param_s, param_t in zip(self.norm.parameters(), self.norm_teacher.parameters()):
            if param_s.requires_grad:
                param_t.data = param_t.data * momentum + param_s.detach().data * (1. - momentum)
        
        # ✅ 优化：更新全局特征投影层
        for param_s, param_t in zip(self.global_proj.parameters(), self.global_proj_teacher.parameters()):
            if param_s.requires_grad:
                param_t.data = param_t.data * momentum + param_s.detach().data * (1. - momentum)

    def forward_encoder_student(self, neighborhood, center, mask):
        """
        Student encoder: 对可见的 patches 进行编码
        """
        B = neighborhood.shape[0]
        
        # 先编码 patches
        group_tokens = self.encoder(neighborhood)  # B G C
        
        # Early Masking: 在编码后立即 mask
        x_vis = group_tokens[~mask].reshape(B, -1, self.trans_dim)  # B N_vis C
        
        # 获取可见 patches 的中心和位置编码
        center_vis = center[~mask].reshape(B, -1, 3)  # B N_vis 3
        pos_vis = self.pos_embed(center_vis)  # B N_vis C
        
        # Transformer blocks
        x = self.blocks(x_vis, pos_vis)  # B N_vis C
        x = self.norm(x)  # B N_vis C
        
        # ✅ 优化：使用 max + mean pooling 的拼接，保留更多信息
        x_mean = x.mean(dim=1)  # B C
        x_max = x.max(dim=1)[0]  # B C
        x_global = torch.cat([x_mean, x_max], dim=-1)  # B 2C
        # 如果维度不匹配，使用投影层（在 __init__ 中添加）
        if hasattr(self, 'global_proj'):
            x_global = self.global_proj(x_global)  # B C
        
        return x, x_global, center_vis

    @torch.no_grad()
    def forward_encoder_teacher(self, neighborhood, center):
        """
        Teacher encoder: 对完整点云进行编码
        """
        # 先编码 patches
        group_tokens = self.encoder_teacher(neighborhood)  # B G C
        
        # Teacher 不使用 mask，直接处理所有 patches
        pos = self.pos_embed_teacher(center)  # B G C
        
        # Transformer blocks
        x = self.blocks_teacher(group_tokens, pos)  # B G C
        x = self.norm_teacher(x)  # B G C
        
        # ✅ 优化：Teacher 也使用 max+mean pooling（与 student 一致）
        x_mean = x.mean(dim=1)  # B C
        x_max = x.max(dim=1)[0]  # B C
        x_global = torch.cat([x_mean, x_max], dim=-1)  # B 2C
        x_global = self.global_proj_teacher(x_global)  # B C
        
        # 返回 patch-level 特征（用于 codebook assignment）和全局特征
        # 注意：这里返回的是 patch tokens，不是全局特征
        return x.contiguous()

    def forward_decoder(self, x_vis, center_vis, center, mask):
        """Decoder: 预测被 mask 的 patches 的 assignment"""
        B, _, C = x_vis.shape
        
        # x_vis 已经是 patch tokens，直接使用
        x_vis_patches = x_vis
        
        # 准备 mask tokens
        center_mask = center[mask].reshape(B, -1, 3)
        num_mask = center_mask.shape[1]
        mask_tokens = self.mask_token.expand(B, num_mask, -1)
        
        # 拼接 visible 和 mask tokens
        x_full = torch.cat([x_vis_patches, mask_tokens], dim=1)
        
        # Position embeddings
        pos_vis = self.decoder_pos_embed(center_vis)
        pos_mask = self.decoder_pos_embed(center_mask)
        pos_full = torch.cat([pos_vis, pos_mask], dim=1)
        
        # Decoder transformer
        x_decoded = self.decoder_blocks(x_full, pos_full)
        
        # 只返回 mask 部分的特征
        x_mask_decoded = x_decoded[:, -num_mask:, :]  # B num_mask C
        
        return x_mask_decoded, center_mask

    def extract_targets(self, pts1, pts2, momentum, update_teacher=True):
        """
        使用 teacher 提取目标 assignments
        参考 moca.py 的实现：teacher 更新在 extract_targets 内部进行
        """
        with torch.no_grad():
            if update_teacher:
                self.update_teacher(momentum)  # ✅ 修复：在 extract_targets 内部更新 teacher（与 moca.py 一致）
            
            # 获取当前 codebook
            dictionary = self.bow_extractor.get_dictionary()
            
            # Teacher 编码两个视图（使用直接的特征提取方式：encoder + blocks）
            neighborhood1, center1 = self.group_divider(pts1)
            neighborhood2, center2 = self.group_divider(pts2)
            
            teacher_features1 = self.forward_encoder_teacher(neighborhood1, center1)
            teacher_features2 = self.forward_encoder_teacher(neighborhood2, center2)
            
            # 提取 BoW 和 codes
            bow_code_x1, codes_x1 = self.bow_extractor([teacher_features1])
            bow_code_x2, codes_x2 = self.bow_extractor([teacher_features2])
        
        # ✅ 修复：codebook 更新策略与 moca.py 一致（每个 batch 都更新，50% 概率随机选择视图）
        if self.training and update_teacher:
            # Update the teacher's codebook / dictionary (与 moca.py 一致)
            if random.random() < 0.5:
                self.bow_extractor.update([teacher_features1])
            else:
                self.bow_extractor.update([teacher_features2])
        
        # Cross-view targets: view1 预测 view2 的 BoW，反之亦然
        same_view_codes = torch.cat([codes_x1[0], codes_x2[0]], dim=0)
        cross_view_bows = torch.cat([bow_code_x2[0], bow_code_x1[0]], dim=0)
        
        return cross_view_bows, same_view_codes, dictionary

    def forward_img_loss(self, target, pred):
        """全局 BoW 预测损失（Cross-view）"""
        # ✅ 修复：添加温度缩放，防止分布过于尖锐导致损失过小
        # target: (B, K) - soft assignment probabilities
        # pred: (B, K) - logits
        
        # 添加温度缩放，使预测分布更平滑（防止过于尖锐）
        temperature = 1.0  # 可以调整，1.0 表示不缩放
        pred_scaled = pred / temperature
        pred_log_softmax = F.log_softmax(pred_scaled, dim=1)
        
        # 添加数值稳定性：避免 target 中有 0 导致 log(0)
        target_smooth = target + 1e-8
        target_smooth = target_smooth / target_smooth.sum(dim=1, keepdim=True)
        
        # ✅ 修复：添加最小损失阈值，防止损失过小
        loss = F.kl_div(pred_log_softmax, target_smooth, reduction="batchmean", log_target=False)
        
        # ✅ 修复：如果损失过小，可能是分布过于尖锐，添加熵正则项
        # 计算预测分布的熵，如果熵过小（分布过于尖锐），增加惩罚
        pred_probs = F.softmax(pred_scaled, dim=1)
        entropy = -(pred_probs * pred_log_softmax).sum(dim=1).mean()
        min_entropy = 1.0  # 最小熵阈值（对于 K=4096，均匀分布的熵约为 log(4096) ≈ 8.3）
        if entropy < min_entropy:
            # 分布过于尖锐，添加熵正则项（增加系数，使惩罚更明显）
            entropy_penalty = 0.1 * (min_entropy - entropy)
            loss = loss + entropy_penalty
        
        # ✅ 修复：如果熵过小且损失过小，可能是分布过于尖锐，增加最小损失
        # 注意：只在熵过小时才添加最小损失，避免影响正常学习
        min_loss = torch.tensor(1e-4, device=loss.device, dtype=loss.dtype)
        if entropy < min_entropy:
            # 分布过于尖锐时，确保损失不会过小
            loss = torch.clamp(loss, min=min_loss)
        
        return loss

    def forward_loc_loss(self, target, pred):
        """局部 patch assignment 损失（Same-view）"""
        # target: (N, K) or (B, G, K)
        # pred: (N, K)
        if target.dim() == 3:
            # 如果 target 是 3D，展平它
            K = target.shape[2]
            target = target.view(-1, K)
        
        # ✅ 修复：添加温度缩放，防止分布过于尖锐导致损失过小
        temperature = 1.0  # 可以调整，1.0 表示不缩放
        pred_scaled = pred / temperature
        pred_log_softmax = F.log_softmax(pred_scaled, dim=1)
        
        # 添加数值稳定性
        target_smooth = target + 1e-8
        target_smooth = target_smooth / target_smooth.sum(dim=1, keepdim=True)
        loss = F.kl_div(pred_log_softmax, target_smooth, reduction="batchmean", log_target=False)
        
        # ✅ 修复：如果损失过小，可能是分布过于尖锐，添加熵正则项
        # 计算预测分布的熵，如果熵过小（分布过于尖锐），增加惩罚
        pred_probs = F.softmax(pred_scaled, dim=1)
        entropy = -(pred_probs * pred_log_softmax).sum(dim=1).mean()
        min_entropy = 1.0  # 最小熵阈值
        if entropy < min_entropy:
            # 分布过于尖锐，添加熵正则项（增加系数，使惩罚更明显）
            entropy_penalty = 0.1 * (min_entropy - entropy)
            loss = loss + entropy_penalty
        
        # ✅ 修复：如果熵过小且损失过小，可能是分布过于尖锐，增加最小损失
        # 注意：只在熵过小时才添加最小损失，避免影响正常学习
        min_loss = torch.tensor(1e-4, device=loss.device, dtype=loss.dtype)
        if entropy < min_entropy:
            # 分布过于尖锐时，确保损失不会过小
            loss = torch.clamp(loss, min=min_loss)
        
        return loss

    def _mask_center_rand(self, center, noaug=False):
        """随机 mask 策略"""
        B, G, _ = center.shape
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool().to(center.device)
        
        num_mask = int(self.mask_ratio * G)
        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - num_mask),
                np.ones(num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        return overall_mask.to(center.device)

    def _mask_center_block(self, center, noaug=False):
        """块状 mask 策略"""
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool().to(center.device)
        
        mask_idx = []
        for points in center:
            points = points.unsqueeze(0)
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1)
            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]
            mask_num = int(self.mask_ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())
        
        bool_masked_pos = torch.stack(mask_idx).to(center.device)
        return bool_masked_pos
    def extract_targets_pregrouped(self, neigh1, cen1, neigh2, cen2, momentum, update_teacher=True):
        with torch.no_grad():
            if update_teacher:
                self.update_teacher(momentum)
            
            dictionary = self.bow_extractor.get_dictionary()
            
            # 直接使用传入的 neighborhood 和 center，不再重新 group
            teacher_features1 = self.forward_encoder_teacher(neigh1, cen1)
            teacher_features2 = self.forward_encoder_teacher(neigh2, cen2)
            
            # ... [后续提取 BoW 和 codes 的逻辑保持不变] ...
            bow_code_x1, codes_x1 = self.bow_extractor([teacher_features1])
            bow_code_x2, codes_x2 = self.bow_extractor([teacher_features2])
            
            # Update Codebook logic ...
            # Return targets ...
            same_view_codes = torch.cat([codes_x1[0], codes_x2[0]], dim=0)
            cross_view_bows = torch.cat([bow_code_x2[0], bow_code_x1[0]], dim=0)
            
            return cross_view_bows, same_view_codes, dictionary
    def forward(self, pts, vis=False, **kwargs):
        """
        前向传播
        
        训练模式：
            pts: 点云数据 [B, N, 3]
            返回 loss
        
        可视化模式：
            vis=True
            返回可视化数据
        """
        if not self.training or vis:
            # 推理或可视化模式
            neighborhood, center = self.group_divider(pts)
            mask = self._mask_center_rand(center) if self.mask_type == 'rand' else self._mask_center_block(center)
            
            # 使用直接的特征提取方式：encoder + blocks
            x_vis, x_global, center_vis = self.forward_encoder_student(neighborhood, center, mask)
            
            if vis:
                # 返回可视化数据（类似 Point-MAE）
                B = pts.shape[0]
                M = mask.sum(dim=1)[0].item()
                vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                full_vis = vis_points + center[~mask].reshape(-1, 1, 3)
                return full_vis.reshape(B, -1, 3), center
            else:
                return x_global
        
        # 训练模式
        momentum = kwargs.get('momentum', 0.996)
        img_weight = kwargs.get('img_weight', 1.0)
        loc_weight = kwargs.get('loc_weight', 1.0)
        
        # ✅ 关键修改：生成两个经过不同几何变换的视图 (Augmented Views)
        pts_view1 = augment_point_cloud(pts)
        pts_view2 = augment_point_cloud(pts)
        
        # 对两个视图分别进行 Tokenization (FPS + KNN)
        neighborhood1, center1 = self.group_divider(pts_view1)
        neighborhood2, center2 = self.group_divider(pts_view2)
        
        # 生成 Mask (Masking)
        mask1 = self._mask_center_rand(center1) if self.mask_type == 'rand' else self._mask_center_block(center1)
        mask2 = self._mask_center_rand(center2) if self.mask_type == 'rand' else self._mask_center_block(center2)
        
        # 确保两个 mask 不完全相同（至少有一些差异）
        max_attempts = 10
        attempt = 0
        while attempt < max_attempts and torch.allclose(mask1.float(), mask2.float()):
            # [Fix 1] 这里原来写的是 center，改为 center2
            mask2 = self._mask_center_rand(center2) if self.mask_type == 'rand' else self._mask_center_block(center2)
            attempt += 1
        
        # 提取 teacher targets
        cross_view_bows, same_view_codes, dictionary = self.extract_targets_pregrouped(
            neighborhood1, center1, neighborhood2, center2, momentum, update_teacher=True
        )
        
        # Student 编码两个被 Mask 的增强视图
        x_vis1, x_global1, center_vis1 = self.forward_encoder_student(neighborhood1, center1, mask1)
        x_vis2, x_global2, center_vis2 = self.forward_encoder_student(neighborhood2, center2, mask2)
        
        # 拼接两个 views
        x_global_both = torch.cat([x_global1, x_global2], dim=0)
        
        # ====== 全局 BoW 预测损失（Cross-view）======
        bow_preds = self.encoder_pred(x_global_both, dictionary)
        loss_img = [self.forward_img_loss(cross_view_bows, pred) for pred in bow_preds]
        loss_img = torch.stack(loss_img).mean() * 2
        loss_tot = loss_img * img_weight
        
        # ====== 局部 patch assignment 损失（Same-view）======
        if self.use_loc_loss:
            # Decoder 预测 masked patches 的 assignment
            # [Fix 2] 将 center 替换为 center1 和 center2
            x_mask_decoded1, _ = self.forward_decoder(x_vis1, center_vis1, center1, mask1)
            x_mask_decoded2, _ = self.forward_decoder(x_vis2, center_vis2, center2, mask2)
            
            # 从 same_view_codes 中提取 masked patches 的 target codes
            B = pts.shape[0]
            codes_view1 = same_view_codes[:B]  # (B, G, K)
            codes_view2 = same_view_codes[B:]  # (B, G, K)
            
            # 使用 mask 选择被遮挡的 patches
            target_codes1 = codes_view1.reshape(-1, codes_view1.shape[-1])[mask1.reshape(-1)]
            target_codes2 = codes_view2.reshape(-1, codes_view2.shape[-1])[mask2.reshape(-1)]
            
            # 拼接预测和目标
            x_mask_decoded = torch.cat([
                x_mask_decoded1.reshape(-1, self.trans_dim),
                x_mask_decoded2.reshape(-1, self.trans_dim)
            ], dim=0)
            
            target_codes_masked = torch.cat([target_codes1, target_codes2], dim=0)
            
            # 预测
            codes_preds = self.decoder_pred(x_mask_decoded, dictionary)
            
            # 计算损失
            loss_loc = [self.forward_loc_loss(target_codes_masked, pred) for pred in codes_preds]
            loss_loc = torch.stack(loss_loc).mean() * 2
        else:
            loss_loc = torch.zeros_like(loss_img)
        
        loss_tot += (loss_loc * loc_weight)
        
        # 统计信息
        if hasattr(self, '_iter_count'):
            self._iter_count += 1
        else:
            self._iter_count = 0
        
        if self._iter_count % 100 == 0:
            print_log(f'[Point_MOCA] Iter {self._iter_count}: loss_img={loss_img.item():.4f}, '
                      f'loss_loc={loss_loc.item():.4f}, loss_tot={loss_tot.item():.4f}',
                      logger='Point_MOCA')
        
        return loss_tot

