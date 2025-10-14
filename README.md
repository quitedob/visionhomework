# OpenCV图像处理实验项目

这是一个基于Python的OpenCV图像处理实验项目，提供交互式的图像处理学习体验。

**作者**: dobqop999@gmail.com  
**GitHub**: [https://github.com/quitedob/visionhomework](https://github.com/quitedob/visionhomework)

## 项目结构

```
visiontest/
├── .gitignore            # Git忽略文件配置
├── .venv/                 # 虚拟环境目录
├── Test1.py              # OpenCV图像处理实验主程序
├── test2.py              # PyTorch MNIST分类实验程序（GUI版本）
├── test3.py              # PyTorch迁移学习猫狗分类实验程序
├── requirements.txt      # 项目依赖
├── test_dependencies.py  # 依赖测试脚本
└── README.md            # 项目说明
```

## 环境配置

### PyTorch安装

在运行PyTorch相关实验前，请确保正确安装PyTorch。您可以访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 根据您的系统配置选择合适的安装方式。

### 使用虚拟环境

项目已配置虚拟环境，包含以下依赖库：

- **OpenCV (opencv-python)** - 计算机视觉库
- **NumPy** - 数值计算库
- **Pillow (PIL)** - 图像处理库
- **PyTorch (torch)** - 深度学习框架
- **TorchVision (torchvision)** - PyTorch计算机视觉库
- **Matplotlib** - 数据可视化库

### 激活虚拟环境

#### Windows:
```bash
# 激活虚拟环境
.venv\Scripts\activate

# 退出虚拟环境
deactivate
```

#### Linux/Mac:
```bash
# 激活虚拟环境
source .venv/bin/activate

# 退出虚拟环境
deactivate
```

## 运行程序

激活虚拟环境后，可以运行以下程序：

### OpenCV图像处理实验
```bash
python Test1.py
```

### PyTorch MNIST分类实验（GUI版本）
```bash
python test2.py
```
*启动现代化的图形界面，支持步骤化操作和实时进度显示*

### PyTorch迁移学习猫狗分类实验
```bash
python test3.py
```

## 功能特性

### OpenCV图像处理实验 (Test1.py)
- 交互式图像处理实验演示
- 支持中英文界面切换
- 步骤化学习引导
- 图像读取、保存和格式对比
- RGB通道分离和合并
- 色彩空间转换演示

### PyTorch MNIST分类实验 (test2.py)
- 基于卷积神经网络的MNIST手写数字识别
- 现代化的Tkinter GUI界面
- 步骤化交互式实验引导
- 实时训练进度显示和状态更新
- 数据预览和模型架构可视化
- 预测结果对比显示
- **批标准化加速收敛**
- **学习率调度器优化训练**
- **tqdm进度条实时显示**

### PyTorch迁移学习猫狗分类实验 (test3.py)
- 使用预训练ResNet50进行迁移学习
- 猫狗二分类任务
- **微调策略：解冻顶层网络，差异化学习率**
- **ReduceLROnPlateau学习率调度器**
- **增强数据增强：旋转、颜色抖动等**
- 完整的训练和测试流程
- 预测结果可视化（修复颜色显示问题）
- **tqdm进度条实时显示训练进度**

## 实验步骤

1. **选择图片** - 选择要处理的图像文件
2. **图像读取和显示** - 显示原始图像基本属性
3. **图像保存 - 质量对比** - JPEG不同质量设置对比
4. **图像保存 - 压缩对比** - PNG不同压缩级别对比
5. **分离灰度图** - 灰度图像分离演示
6. **图像创建和复制** - 创建新图像和复制操作
7. **RGB通道分离** - 分离并显示RGB颜色通道
8. **通道合并与色彩空间** - 合并通道和BGR与RGB色彩空间差异

### PyTorch MNIST分类实验步骤

1. **数据准备** - 下载并预处理MNIST数据集，分割为训练集和测试集
2. **数据装载** - 创建DataLoader进行批次处理
3. **数据预览** - 可视化训练数据样本
4. **模型搭建** - 构建卷积神经网络（CNN）架构
5. **模型训练** - 使用Adam优化器和交叉熵损失进行训练
6. **模型测试** - 评估模型性能并可视化预测结果

### PyTorch MNIST分类实验步骤（GUI版本）

1. **加载数据** - 自动下载并加载MNIST数据集
2. **数据预览** - 可视化训练数据样本和标签
3. **搭建模型** - 构建卷积神经网络并显示模型架构
4. **训练模型** - 实时显示训练进度和准确率
5. **测试模型** - 评估模型性能并可视化预测结果

### PyTorch迁移学习猫狗分类实验步骤（增强版）

1. **数据准备** - 下载并预处理猫狗数据集，使用增强数据变换
2. **模型迁移** - 自动下载预训练ResNet50模型
3. **模型修改** - 微调策略：解冻layer4，使用差异化学习率
4. **模型训练** - 使用ReduceLROnPlateau调度器动态调整学习率
5. **模型测试** - 评估模型性能并可视化预测结果（颜色修正）

## 依赖库说明

所有必需的库都已安装在虚拟环境中：

### OpenCV实验依赖库
- `opencv-python` - OpenCV计算机视觉库
- `numpy` - 数值计算库
- `Pillow` - Python图像处理库

### PyTorch实验依赖库
- `torch` - PyTorch深度学习框架
- `torchvision` - PyTorch计算机视觉库（包含MNIST数据集等）
- `matplotlib` - 数据可视化库
- `Pillow` - 图像处理库（用于GUI图像显示）
- `tqdm` - 进度条库（用于训练进度显示）

tkinter和os是Python内置库，无需额外安装。


## 数据集准备

### 猫狗分类实验数据集

实验3需要猫狗分类数据集，请按以下结构组织数据：

```
data/DogsVSCats/
├── train/
│   ├── cat/     # 猫的训练图片 (jpg/png格式)
│   └── dog/     # 狗的训练图片 (jpg/png格式)
└── valid/
    ├── cat/     # 猫的验证图片 (jpg/png格式)
    └── dog/     # 狗的验证图片 (jpg/png格式)
```

您可以从[Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)下载数据集。


## 注意事项

- 虚拟环境已配置完成，无需手动安装依赖
- 确保在虚拟环境中运行程序以获得正确的依赖版本
- 如需重新安装依赖，可运行：`pip install -r requirements.txt`
- 实验3首次运行时会自动下载ResNet50预训练模型，需要网络连接
