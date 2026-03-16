import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from logger import get_missing_parameters_message, get_unexpected_parameters_message

from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from pointnet2_utils import PointNetFeaturePropagation


def fps(data, number):
    '''
    最远点采样(Furthest Point Sampling)函数
    参数:
        data: B N 3 - B批次大小，N点数量，3坐标维度
        number: int - 要采样的点数量
    返回:
        fps_data: B number 3 - 采样后的点云
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)  # 获取最远点采样的索引
    # gather_operation将原始点云中的对应点收集起来
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        '''
        点云分组模块
        参数:
            num_group: 分组数量G
            group_size: 每组内的点数量M
        '''
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)  # KNN搜索模块

    def forward(self, xyz):
        '''
        输入: B N 3 - 原始点云
        输出:
            neighborhood: B G M 3 - 分组后的局部点云
            center: B G 3 - 每个组的中心点
        '''
        batch_size, num_points, _ = xyz.shape
        # 使用FPS采样获取中心点
        center = fps(xyz, self.num_group)  # B G 3
        # 使用KNN获取每个中心点周围的邻域点
        _, idx = self.knn(xyz, center)  # B G M - 每个中心点的K个最近邻索引
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size

        # 处理批次索引
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        # 收集邻域点
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        # 将邻域点坐标标准化为相对于中心点的局部坐标
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        '''
        点云局部特征编码器
        参数:
            encoder_channel: 输出特征维度
        '''
        super().__init__()
        self.encoder_channel = encoder_channel

        # 第一层卷积网络，将3D坐标转换为高维特征
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        # 第二层卷积网络，结合全局和局部特征
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
        输入: point_groups - B G N 3 (批次，组数，每组点数，坐标维度)
        输出: feature_global - B G C (批次，组数，特征维度)
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)  # 将所有组展平处理

        # 编码器前向传播
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 N

        # 获取全局特征并扩展
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1 - 最大池化

        # 拼接全局和局部特征
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 N

        # 第二层编码
        feature = self.second_conv(feature)  # BG C N

        # 再次全局池化获得每个组的特征表示
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG C

        return feature_global.reshape(bs, g, self.encoder_channel)  # B G C


class Mlp(nn.Module):
    '''
    多层感知机模块，用于Transformer中的FFN(Feed Forward Network)
    '''

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一个全连接层
        self.act = act_layer()  # 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二个全连接层
        self.drop = nn.Dropout(drop)  # Dropout正则化

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    '''
    多头自注意力机制模块
    '''

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个头的维度

        # 注意力缩放因子
        self.scale = qk_scale or head_dim ** -0.5

        # QKV线性投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # 输出投影
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        '''
        输入: x - B N C (批次，序列长度，特征维度)
        '''
        B, N, C = x.shape

        # 线性投影并分离Q、K、V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别获取查询、键、值

        # 计算注意力得分
        attn = (q * self.scale) @ k.transpose(-2, -1)  # B H N N
        attn = attn.softmax(dim=-1)  # Softmax归一化
        attn = self.attn_drop(attn)  # Dropout

        # 应用注意力得分
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    '''
    Transformer编码器块
    '''

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        # 第一个层归一化
        self.norm1 = norm_layer(dim)

        # Dropout路径，用于随机丢弃整个块的输出
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 第二个层归一化
        self.norm2 = norm_layer(dim)

        # MLP前馈网络
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 自注意力层
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        # 残差连接 + 自注意力
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # 残差连接 + MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    '''
    Transformer编码器，包含多个编码器块
    '''

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        # 创建多个编码器块
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                # 随着深度增加逐渐增大drop_path_rate
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        '''
        输入:
            x: 特征向量
            pos: 位置编码
        返回:
            feature_list: 特定层的特征列表
        '''
        feature_list = []
        fetch_idx = [3, 7, 11]  # 选择提取特征的层索引

        for i, block in enumerate(self.blocks):
            x = block(x + pos)  # 添加位置编码
            if i in fetch_idx:
                feature_list.append(x)  # 收集特定层的特征

        return feature_list


class get_model(nn.Module):
    '''
    完整的点云语义分割模型 - 适配S3DIS数据集
    '''

    def __init__(self, cls_dim):
        '''
        参数:
            cls_dim: 分类维度，S3DIS数据集为13个类别
        '''
        super().__init__()

        # 模型参数设置
        self.trans_dim = 384  # Transformer特征维度
        self.depth = 12  # Transformer深度
        self.drop_path_rate = 0.1  # DropPath率
        self.cls_dim = cls_dim  # 分类维度 - S3DIS为13
        self.num_heads = 6  # 注意力头数

        # 分组参数
        self.group_size = 32  # 每组点数
        self.num_group = 256  # 组数量 - 增加以适应更大的点云

        # 点云分组模块
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # 定义特征编码器
        self.encoder_dims = 384
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        # 位置编码，将3D坐标映射到高维特征
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        # 输入特征转换 - 处理S3DIS数据的9维特征(xyz, rgb, 归一化xyz)
        self.input_trans = nn.Sequential(
            nn.Conv1d(9, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        # 创建递减的DropPath率列表
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        # Transformer编码器
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        # 层归一化
        self.norm = nn.LayerNorm(self.trans_dim)

        # 特征传播层，用于将特征上采样回原始点云
        self.propagation_0 = PointNetFeaturePropagation(
            in_channel=1152 + 3,  # 3个Transformer特征层拼接(384*3)加上坐标
            mlp=[self.trans_dim * 4, 1024]
        )

        # 分类头
        # 直接连接全局特征和局部特征，不使用类别标签
        self.convs1 = nn.Conv1d(1024 + 1152 * 2, 512, 1)  # 3328 = 1024 + 1152*2
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def get_loss_acc(self, ret, gt, weights=None):
        '''
        计算损失和准确率
        '''
        if weights is not None:
            loss = F.nll_loss(ret, gt.long(), weight=weights)
        else:
            loss = F.nll_loss(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        '''
        从预训练检查点加载模型参数
        '''
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path, map_location='cpu', weights_only=False)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            # 处理不同命名空间的键
            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            # 加载参数
            incompatible = self.load_state_dict(base_ckpt, strict=False)

            # 打印缺失和意外的参数
            if incompatible.missing_keys:
                print('missing_keys')
                print(get_missing_parameters_message(incompatible.missing_keys))
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(get_unexpected_parameters_message(incompatible.unexpected_keys))

            print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def forward(self, pts):
        '''
        模型前向传播 - 为S3DIS数据集修改
        参数:
            pts: B 9 N - S3DIS数据集点云(批次，特征维度9，点数)
                 9维特征: 原始xyz(3), RGB颜色(3), 归一化xyz坐标(3)
        返回:
            x: B N cls_dim - 每个点的类别预测分数
        '''
        B, C, N = pts.shape

        # 提取点云坐标和特征
        xyz = pts[:, 0:3, :].transpose(-1, -2).contiguous()  # B N 3
        features = pts.clone()  # B 9 N

        # 将点云分组
        neighborhood, center = self.group_divider(xyz)  # B G M 3, B G 3

        # 编码每个组的特征
        group_input_tokens = self.encoder(neighborhood)  # B G C

        # 计算位置编码
        pos = self.pos_embed(center)  # B G C

        # Transformer输入
        x = group_input_tokens

        # 通过Transformer编码器
        feature_list = self.blocks(x, pos)  # 获取多层特征

        # 归一化并调整维度
        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]  # B C G

        # 拼接多层特征
        x = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1)  # B 1152 G

        # 全局特征提取 - 最大值和平均值
        x_max = torch.max(x, 2)[0]  # B 1152
        x_avg = torch.mean(x, 2)  # B 1152

        # 扩展全局特征到每个点
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)  # B 1152 N
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)  # B 1152 N

        # 特征传播 - 将特征上采样到原始点云
        f_level_0 = self.propagation_0(
            xyz.transpose(-1, -2),
            center.transpose(-1, -2),
            xyz.transpose(-1, -2),
            x
        )  # B 1024 N

        # 拼接所有特征 - 不使用类别标签特征
        x = torch.cat((f_level_0, x_max_feature, x_avg_feature), 1)  # B (1024+1152+1152) N

        # 通过分类头
        x = self.relu(self.bns1(self.convs1(x)))  # B 512 N
        x = self.dp1(x)  # dropout
        x = self.relu(self.bns2(self.convs2(x)))  # B 256 N
        x = self.convs3(x)  # B cls_dim N

        # 应用log_softmax进行多分类
        x = F.log_softmax(x, dim=1)

        # 调整维度顺序
        x = x.permute(0, 2, 1)  # B N cls_dim

        return x


class get_loss(nn.Module):
    '''
    损失函数模块 - 使用负对数似然损失，支持权重
    '''

    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat=None, weights=None):
        '''
        参数:
            pred: B N cls_dim - 预测分数
            target: B N - 目标类别索引
            trans_feat: 未使用
            weights: 类别权重
        '''
        if weights is not None:
            weights = weights.float().cuda()
            total_loss = F.nll_loss(pred, target, weight=weights)
        else:
            total_loss = F.nll_loss(pred, target)

        return total_loss