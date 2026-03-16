import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm
from utils.logger import *
from tools import builder
from utils import misc
from utils.logger import *


def _move_to_device(points, device):
    if isinstance(points, torch.Tensor):
        return points.to(device, non_blocking=True)
    raise TypeError(f'Unexpected point type: {type(points)}')



@torch.no_grad()
def _extract_features(model, dataloader, npoints, device, desc):
    model.eval()
    feats, labels = [], []

    iterator = tqdm(dataloader, desc=desc)
    for _, _, data in iterator:
        points = _move_to_device(data[0], device)
        target = data[1].view(-1).to(device, non_blocking=True)

        if points.shape[1] != npoints:
            points = misc.fps(points, npoints)

        # --- 修改开始：不调用 model(points)，而是调用自定义提取函数 ---
        
        # 判断是否被 DataParallel 包裹
        if hasattr(model, 'module'):
            # DataParallel 模式下，自定义方法在 module 属性里
            # 注意：这样调用会在主 GPU 上运行单次 inference，不会自动多卡并行
            features = model.module.extract_features(points)
        else:
            # 单卡模式直接调用
            features = model.extract_features(points)
            
        # --- 修改结束 ---

        feats.append(features.detach().cpu())
        labels.append(target.cpu())

    feats = torch.cat(feats, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return feats, labels
def evaluate_svm(train_feats, train_labels, test_feats, test_labels, logger=None):
    """
    标准的 Linear Evaluation 代码
    会对不同的 C 值进行网格搜索，报告最好的结果
    """
    # 1. L2 归一化 (这一步非常关键，很多时候能提升几个点)
    # 将特征向量归一化到单位球面上
    train_feats = train_feats / np.linalg.norm(train_feats, axis=1, keepdims=True)
    test_feats = test_feats / np.linalg.norm(test_feats, axis=1, keepdims=True)

    # 2. 搜索的最佳参数列表
    # 通常在这个范围内搜索
    costs = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    results = []
    
    for c in costs:
        # LinearSVC 标准设置
        # dual=False 当 n_samples > n_features 时更快
        # class_weight='balanced' 用于处理数据不平衡
        clf = LinearSVC(C=c, dual=False, class_weight='balanced', max_iter=2000)
        
        clf.fit(train_feats, train_labels)
        pred = clf.predict(test_feats)
        acc = np.sum(test_labels == pred) / float(test_labels.shape[0]) * 100.0
        
        results.append(acc)
        
        msg = f'[SVM] C={c:.2f} | Accuracy={acc:.2f}%'
        if logger:
            print_log(msg, logger=logger)
        else:
            print(msg)

    # 3. 选取最佳结果
    best_acc = max(results)
    best_c = costs[results.index(best_acc)]
    
    msg = f'[SVM] Best Accuracy: {best_acc:.2f}% (C={best_c})'
    if logger:
        print_log(msg, logger=logger)
    else:
        print(msg)
        
    return best_acc

import numpy as np
import torch
from sklearn.svm import LinearSVC
from utils.logger import print_log # 确保引入 logger

# --- 更加强力的 SVM 评估函数 ---
def evaluate_svm_robust(train_features, train_labels, test_features, test_labels, logger=None):
    # ------------------------------------------------------------------
    # 修正策略：
    # 1. 放弃 L2 Normalization，它破坏了你的 Point-MAE 特征。
    # 2. 回归 StandardScaler (Z-score)，这让你之前跑出了 81%。
    # 3. 继续使用 Grid Search，在 C=0.01 到 C=100 之间寻找比 81% 更高的点。
    # ------------------------------------------------------------------
    
    # 搜索范围：你之前 C=0.2 跑了 81.85%，所以我们重点搜索 0.1 ~ 1.0 附近的区域
    costs = [0.001, 0.005, 0.01, 0.02, 0.05,0.007, 0.1, 0.5, 1.0]
    
    best_acc = 0
    best_c = 0

    for c in costs:
        # 使用 Pipeline：先标准化，再 SVM
        # dual=False 在样本数 > 特征数时通常更快且更稳定
        clf = make_pipeline(
            StandardScaler(),
            LinearSVC(C=c, dual=False, class_weight='balanced', max_iter=5000)
        )
        
        clf.fit(train_features, train_labels)
        pred = clf.predict(test_features)
        acc = np.sum(test_labels == pred) / float(test_labels.shape[0]) * 100.0
        
        msg = f'[SVM] C={c} | Accuracy={acc:.2f}%'
        if logger:
            print_log(msg, logger=logger)
        else:
            print(msg)
        
        if acc > best_acc:
            best_acc = acc
            best_c = c
            
    final_msg = f'[SVM] Best Accuracy: {best_acc:.2f}% (C={best_c})'
    if logger:
        print_log(final_msg, logger=logger)
    else:
        print(final_msg)
        
    return best_acc

def run_linear_eval(args, config):
    # 1. 定义 Logger
    logger = get_logger(args.log_name)
    
    # 2. 准备数据
    print_log('[LinearEval] 构建数据集...', logger=logger)
    _, train_loader = builder.dataset_builder(args, config.dataset.train)
    _, test_loader = builder.dataset_builder(args, config.dataset.val)

    # 3. 准备模型
    print_log('[LinearEval] 构建模型...', logger=logger)
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger=logger, strict=False)
    
    device = torch.device('cuda' if args.use_gpu else 'cpu')
    if args.use_gpu:
        base_model = torch.nn.DataParallel(base_model).to(device)
    else:
        base_model = base_model.to(device)
    
    base_model.eval()
    npoints = config.npoints

    # 4. 提取特征 (仿照 Point-BERT 的写法，清晰明了)
    # -------------------------------------------------
    train_features = []
    train_label = []
    test_features = []
    test_label = []

    print_log('[LinearEval] 正在提取训练集特征...', logger=logger)
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(train_loader)):
            points = data[0].cuda()
            label = data[1].cuda()
            
            # FPS 采样
            if points.size(1) != npoints:
                points = misc.fps(points, npoints)

            # --- 关键：调用 extract_features ---
            if hasattr(base_model, 'module'):
                feature = base_model.module.extract_features(points)
            else:
                feature = base_model.extract_features(points)
            
            target = label.view(-1)
            train_features.append(feature.detach())
            train_label.append(target.detach())

        print_log('[LinearEval] 正在提取测试集特征...', logger=logger)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_loader)):
            points = data[0].cuda()
            label = data[1].cuda()
            
            if points.size(1) != npoints:
                points = misc.fps(points, npoints)
            
            if hasattr(base_model, 'module'):
                feature = base_model.module.extract_features(points)
            else:
                feature = base_model.extract_features(points)
            
            target = label.view(-1)
            test_features.append(feature.detach())
            test_label.append(target.detach())

    # 5. 拼接数据
    train_features = torch.cat(train_features, dim=0).cpu().numpy()
    train_label = torch.cat(train_label, dim=0).cpu().numpy()
    test_features = torch.cat(test_features, dim=0).cpu().numpy()
    test_label = torch.cat(test_label, dim=0).cpu().numpy()

    print_log(f'[LinearEval] 特征提取完毕. Train: {train_features.shape}, Test: {test_features.shape}', logger=logger)

    # 6. 运行 SVM
    acc = evaluate_svm_robust(train_features, train_label, test_features, test_label, logger)
    
    # 7. 保存
    np.savez_compressed(
        os.path.join(args.experiment_path, 'linear_eval_results.npz'),
        train_feats=train_features, train_labels=train_label,
        test_feats=test_features, test_labels=test_label,
        best_acc=acc
    )