# 自己从0开始的基于YOLOv7的猪只目标识别模型技术文档

## 1. 项目概述
本项目旨在训练一个能够准确识别图像和视频中猪只的高性能目标检测模型。我们选用在速度和精度上均表现出色的YOLOv7算法，并利用Roboflow平台获取并格式化了猪只数据集。

### 1.1 技术栈
- 模型算法: YOLOv7
- 深度学习框架: PyTorch
- 数据集平台: Roboflow
- 核心编程语言: Python

### 1.2 硬件与环境建议
- 操作系统: Linux (Ubuntu 18.04 或更高版本)
- GPU: NVIDIA GPU (推荐 V100, RTX 30系列, A100等)，显存不低于16GB
- 软件依赖:
  - CUDA 11.x
  - cuDNN 8.x
  - Python 3.8+
  - Git

## 2. 环境搭建与项目准备

### 2.1 克隆YOLOv7官方代码库
```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
```

### 2.2 创建并激活Python虚拟环境
```bash
conda create -n yolov7-pigs python=3.8 -y
conda activate yolov7-pigs
```

### 2.3 安装所需的依赖包
```bash
pip install -r requirements.txt
```
> 注意: 请确保你的PyTorch版本与你的CUDA版本相匹配。如果安装缓慢或出错，可以考虑使用国内镜像源。

## 3. 数据集准备

### 3.1 从Roboflow获取数据集
1. 访问 Roboflow 官网并搜索 "Pig" 相关的公开数据集。
2. 选择并下载，导出格式选择 "YOLOv7 PyTorch"。
3. 解压到 `yolov7/data/` 目录下。

### 3.2 data.yaml 配置文件
```yaml
train: ../data/pigs_dataset/train/images
val: ../data/pigs_dataset/valid/images
nc: 1
names: ['pig']
```
- train: 训练集图片路径
- val: 验证集图片路径
- nc: 类别总数
- names: 类别名称列表

## 4. 模型训练

### 4.1 下载预训练权重
```bash
cd yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```
- --weights: 初始权重文件路径
- --data: 数据集配置文件路径
- --cfg: 模型配置文件
- --img-size: 输入图像尺寸
- --batch-size: 批次大小
- --epochs: 训练轮数
- --device: GPU编号或cpu
- --name: 实验名称
- --hyp: 超参数文件路径

训练过程会在终端显示每个epoch的进度，包括损失和评估指标（mAP）。

## 5. 模型评估

使用 `test.py` 进行评估：
```bash
python test.py \
  --weights runs/train/yolov7_pigs_run/weights/best.pt \
  --data data/pig_data.yaml \
  --task val \
  --img-size 640 \
  --device 0
```
评估结果会保存在 `runs/testexp` 目录。

## 6. 模型推理

使用 `detect.py` 进行推理：
```bash
python detect.py \
  --weights runs/train/yolov7_pigs_run/weights/best.pt \
  --source path/to/your/pig_image.jpg \
  --img-size 640 \
  --conf-thres 0.4 \
  --iou-thres 0.5 \
  --device 0
```
推理结果保存在 `runs/detect/exp` 目录。

---
