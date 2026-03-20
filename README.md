# Point-DPA

## Note: If your work uses this algorithm or makes improvements based on it, please be sure to cite this paper. Thank you for your cooperation.
## 注意：如果您的工作用到了本算法，或者基于本算法进行了改进，请您务必引用本论文，谢谢配合
## Point-DPA: Unifying Contrastive and Generative Learning for 3D Point Cloud Understanding via Dynamic Prototypes

Xin Cao,Xinmeng Hu, Yinan Wang, Kang Li , Linzhi Su※ , Yangyang Liu, Fengjun Zhao※ 

Information Sciences Volume 744, 15 July 2026, 123378

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

We use ShapeNet, ScanObjectNN, ModelNet40 and ShapeNetPart in this work. 

## 3. Point-DPA Models
|  Task | Dataset | Acc.|  
|  ----- | ----- |-----|
|  Pre-training | ShapeNet | N.A. |
|  Classification | ScanObjectNN | 87.03%|
|  Classification | ScanObjectNN |92.03% |
|  Classification | ScanObjectNN |89.45%|
|  Classification | ModelNet40(1k) | 93.52%|
| Part segmentation| ShapeNetPart| 85.9% mIoU|



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
@article{cao2026point,
  title={Point-DPA: Unifying contrastive and generative learning for 3D point cloud understanding via dynamic prototypes},
  author={Cao, Xin and Hu, Xinmeng and Wang, Yinan and Li, Kang and Su, Linzhi and Liu, Yangyang and Zhao, Fengjun},
  journal={Information Sciences},
  pages={123378},
  year={2026},
  publisher={Elsevier}
}
```
