import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *


import numpy as np


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


# visualization
def test(base_model, test_dataloader, args, config, logger = None):
    base_model.eval()  # 保留模型eval模式（核心必要逻辑）
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            # 3. 移除：类别筛选（useful_cate）、角度设置（a/b）等可视化相关逻辑
            
            # 保留：数据集判断和设备分配（核心推理逻辑）
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda() if args.use_gpu else data
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            # 4. 保留模型前向推理，但去掉可视化相关的返回值（如果模型支持）
            # 若模型必须传vis=True，可保留；若不需要，改为vis=False
            # 注：如果模型返回值数量不匹配，需根据实际情况调整（比如只取dense_points）
            dense_points, vis_points, centers = base_model(points, vis=False)  # vis改为False
            
            # 5. 移除：所有文件保存、图像生成、cv2写入等可视化代码
            # （包括txt保存、get_ptcloud_img、final_image、cv2.imwrite等）

            # 保留：可选的终止条件（按需保留/删除）
            if idx > 1500:
                break

        return