# Point-DPA

## [Your Paper Title], [Conference/Journal Year] ([ArXiv Link])

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/your-paper-title/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=your-paper-title)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/your-paper-title/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=your-paper-title)

[Brief description of your Point-DPA method. Explain the key contributions and improvements over existing methods. For example: In this work, we present Point-DPA, a novel self-supervised learning method for point clouds that... In classification tasks, Point-DPA outperforms...]

<div  align="center">    
 <img src="./figure/net.jpg" width = "666"  align=center />
</div>

## 1. Requirements
PyTorch >= 1.7.0 < 1.11.0;
python >= 3.7;
CUDA >= 9.0;
GCC >= 4.9;
torchvision;

```
pip install -r requirements.txt
```
<details>
<summary> For Linux Kernel 6.0 or above (e.g. Ubuntu 24)
</summary>

Solution from [Sam Cheung](https://github.com/deemoe404).

Please run the following command before installing Chamfer Distance:
```
sudo apt install gcc-10 g++-10

su
cd /usr/local/src
wget https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.4.tar.xz
tar -xf linux-5.4.tar.xz && cd linux-5.4
make headers_install INSTALL_HDR_PATH=/usr/local/linux-headers-5.4

export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CFLAGS="-I/usr/local/linux-headers-5.4/include"
export CPPFLAGS="-I/usr/local/linux-headers-5.4/include"
```

In `extensions/chamfer_dist/setup.py`, in the `extra_compile_args` field, pass the correct header path to nvcc by adding the following line as the second element of `ext_modules`:
```
extra_compile_args={"nvcc": ['--system-include=/usr/local/linux-headers-5.4/include']}
```

</details>

```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## 2. Datasets

We use ShapeNet, ScanObjectNN, ModelNet40 and ShapeNetPart in this work. See [DATASET.md](../DATASET.md) for details.

## 3. Point-DPA Models
|  Task | Dataset | Config | Acc.| Download|      
|  ----- | ----- |-----|  -----| -----|
|  Pre-training | ShapeNet |[pretrain_dpa.yaml](../cfgs/pretrain_dpa.yaml)| N.A. | [here](https://github.com/your-repo/Point-DPA/releases/download/main/pretrain.pth) |
|  Classification | ScanObjectNN |[finetune_scan_hardest_dpa.yaml](../cfgs/finetune_scan_hardest_dpa.yaml)| XX.XX%| [here](https://github.com/your-repo/Point-DPA/releases/download/main/scan_hardest.pth)  |
|  Classification | ScanObjectNN |[finetune_scan_objbg_dpa.yaml](../cfgs/finetune_scan_objbg_dpa.yaml)|XX.XX% | [here](https://github.com/your-repo/Point-DPA/releases/download/main/scan_objbg.pth) |
|  Classification | ScanObjectNN |[finetune_scan_objonly_dpa.yaml](../cfgs/finetune_scan_objonly_dpa.yaml)| XX.XX%| [here](https://github.com/your-repo/Point-DPA/releases/download/main/scan_objonly.pth) |
|  Classification | ModelNet40(1k) |[finetune_modelnet_dpa.yaml](../cfgs/finetune_modelnet_dpa.yaml)| XX.XX%| [here](https://github.com/your-repo/Point-DPA/releases/download/main/modelnet_1k.pth) |
|  Classification | ModelNet40(8k) |[finetune_modelnet_8k_dpa.yaml](../cfgs/finetune_modelnet_8k_dpa.yaml)| XX.XX%| [here](https://github.com/your-repo/Point-DPA/releases/download/main/modelnet_8k.pth) |
| Part segmentation| ShapeNetPart| [segmentation](../segmentation)| XX.X% mIoU| [here](https://github.com/your-repo/Point-DPA/releases/download/main/part_seg.pth) |

|  Task | Dataset | Config | 5w10s Acc. (%)| 5w20s Acc. (%)| 10w10s Acc. (%)| 10w20s Acc. (%)|     
|  ----- | ----- |-----|  -----| -----|-----|-----|
|  Few-shot learning | ModelNet40 |[fewshot_dpa.yaml](../cfgs/fewshot_dpa.yaml)| XX.X ± X.X| XX.X ± X.X| XX.X ± X.X| XX.X ± X.X| 

## 4. Point-DPA Pre-training
To pretrain Point-DPA on ShapeNet training set, run the following command. If you want to try different models or masking ratios etc., first create a new config file, and pass its path to --config.

```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/pretrain_dpa.yaml --exp_name <output_file_name>
```

## 5. Point-DPA Fine-tuning

Fine-tuning on ScanObjectNN, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_scan_hardest_dpa.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Fine-tuning on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_modelnet_dpa.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Voting on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --test --config cfgs/finetune_modelnet_dpa.yaml \
--exp_name <output_file_name> --ckpts <path/to/best/fine-tuned/model>
```
Few-shot learning, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/fewshot_dpa.yaml --finetune_model \
--ckpts <path/to/pre-trained/model> --exp_name <output_file_name> --way <5 or 10> --shot <10 or 20> --fold <0-9>
```
Part segmentation on ShapeNetPart, run:
```
cd segmentation
python main.py --ckpts <path/to/pre-trained/model> --root path/to/data --learning_rate 0.0002 --epoch 300
```




## Acknowledgements

Our codes are built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) and [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

## Reference

```
@inproceedings{your2024point,
  title={Point-DPA: [Your Paper Title]},
  author={Your Name and Co-authors},
  booktitle={Conference Name},
  pages={XX--XX},
  year={2024},
  organization={Publisher}
}
```
