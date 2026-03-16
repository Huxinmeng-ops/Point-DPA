# 导入必要的库
import os                           # 提供与操作系统交互的功能，用于文件路径处理
import sys                          # 提供系统相关的功能和参数
from indoor3d_util import DATA_PATH, collect_point_label  # 导入自定义模块中的数据路径和点云标签收集函数

# 设置项目路径常量
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 获取当前文件所在的目录的绝对路径
ROOT_DIR = os.path.dirname(BASE_DIR)                    # 获取项目根目录
sys.path.append(BASE_DIR)                              # 将当前目录添加到Python解释器的搜索路径中

# 读取标注路径列表
# 从anno_paths.txt文件中读取每一行，并移除行尾的空白字符
anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
# 将相对路径转换为绝对路径
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

# 创建输出文件夹
output_folder = os.path.join(ROOT_DIR, 'data/stanford_indoor3d')  # 设置输出文件夹路径
if not os.path.exists(output_folder):                            # 如果输出文件夹不存在
    os.mkdir(output_folder)                                      # 创建输出文件夹

# 注意：v1.2版本的数据在Area_5/hallway_6中有一个额外的字符。已手动修复。
# 遍历每个标注文件路径
for anno_path in anno_paths:
    print(anno_path)  # 打印当前处理的标注文件路径
    try:
        # 解析文件路径，提取区域和房间信息
        elements = anno_path.split('/')
        # 构建输出文件名，格式为"区域_房间.npy"，例如：Area_1_hallway_1.npy
        out_filename = elements[-3]+'_'+elements[-2]+'.npy'
        # 调用collect_point_label函数处理点云数据并保存为numpy格式
        collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
    except:
        # 如果处理过程中出现错误，打印错误信息
        print(anno_path, 'ERROR!!')