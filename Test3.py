# 实验3: 迁移学习ResNet50猫狗图像分类 - GUI版本
# 使用预训练ResNet50模型进行猫狗二分类任务 - 交互式界面

import torch  # PyTorch深度学习框架
import torchvision  # PyTorch计算机视觉库
from torchvision import datasets, models, transforms  # 数据集、模型和变换
import os  # 操作系统接口
import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 数据可视化
import time  # 时间测量
from tqdm import tqdm  # 进度条库
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 学习率调度器

# GUI相关导入
import tkinter as tk  # Python标准GUI库
from tkinter import ttk, filedialog, messagebox  # Tkinter子模块
from PIL import Image, ImageTk  # Python图像处理库
import threading  # 多线程支持
import queue  # 线程间通信队列

class ResNet50CatDogGUI:
    """
    ResNet50猫狗分类实验GUI主类

    提供交互式的迁移学习实验演示，包括：
    - 数据集加载和预览
    - 模型配置和迁移学习设置
    - 实时训练进度监控
    - 模型性能分析和可视化
    - 单个图像预测功能

    支持中英文界面切换，现代化UI设计
    """
    def __init__(self):
        """
        初始化ResNet50猫狗分类GUI应用程序

        创建Tkinter主窗口，初始化所有变量和数据结构，
        设置多语言支持，默认使用中文界面
        """
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("ResNet50猫狗分类 - 迁移学习实验")
        self.root.geometry("1600x1000")  # 窗口大小：1600x1000像素
        self.root.resizable(True, True)  # 允许调整窗口大小

        # 设置现代化的窗口样式
        self.root.configure(bg='#f8f9fa')

        # 初始化实例变量
        self.data_dir = ""  # 数据集目录
        self.image_datasets = {}  # 图像数据集
        self.dataloaders = {}  # 数据加载器
        self.model = None  # ResNet50模型
        self.current_language = "zh"  # 当前语言，默认为中文

        # 训练相关变量
        self.is_training = False  # 训练状态标志
        self.training_thread = None  # 训练线程
        self.stop_training = False  # 停止训练标志
        self.training_history = []  # 训练历史记录
        self.current_epoch = 0  # 当前训练轮数
        self.total_epochs = 10  # 总训练轮数
        self.best_accuracy = 0.0  # 最佳准确率
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 计算设备

        # GUI通信队列
        self.message_queue = queue.Queue()

        # 多语言文本资源
        self.text_resources = {
            "zh": {
                "app_title": "ResNet50猫狗分类 - 迁移学习实验",
                "data_tab": "数据设置",
                "model_tab": "模型配置",
                "training_tab": "训练监控",
                "testing_tab": "模型测试",
                "language": "语言",
                "status_ready": "准备开始迁移学习实验。",
                "status_no_data": "请先选择并加载数据集。",
                "status_data_loaded": "数据集加载完成！",
                "status_model_built": "模型构建完成！",
                "status_training": "模型正在训练中...",
                "status_trained": "模型训练完成！",
                "select_dataset": "选择数据集目录",
                "load_data": "加载数据",
                "build_model": "构建模型",
                "start_training": "开始训练",
                "stop_training": "停止训练",
                "test_model": "测试模型",
                "select_image": "选择图片",
                "predict": "预测",
                "batch_size": "批次大小:",
                "epochs": "训练轮数:",
                "learning_rate": "学习率:",
                "fine_tune": "微调模式",
                "feature_extract": "特征提取",
                "use_scheduler": "使用学习率调度器",
                "train_progress": "训练进度",
                "loss_curve": "损失曲线",
                "accuracy_curve": "准确率曲线",
                "dataset_info": "数据集信息",
                "model_info": "模型信息",
                "training_stats": "训练统计",
                "prediction_result": "预测结果",
                "confusion_matrix": "混淆矩阵"
            },
            "en": {
                "app_title": "ResNet50 Cat-Dog Classification - Transfer Learning",
                "data_tab": "Data Setup",
                "model_tab": "Model Config",
                "training_tab": "Training Monitor",
                "testing_tab": "Model Testing",
                "language": "Language",
                "status_ready": "Ready to start transfer learning experiment.",
                "status_no_data": "Please select and load dataset first.",
                "status_data_loaded": "Dataset loaded successfully!",
                "status_model_built": "Model built successfully!",
                "status_training": "Model training in progress...",
                "status_trained": "Model training completed!",
                "select_dataset": "Select Dataset Directory",
                "load_data": "Load Data",
                "build_model": "Build Model",
                "start_training": "Start Training",
                "stop_training": "Stop Training",
                "test_model": "Test Model",
                "select_image": "Select Image",
                "predict": "Predict",
                "batch_size": "Batch Size:",
                "epochs": "Epochs:",
                "learning_rate": "Learning Rate:",
                "fine_tune": "Fine-tuning",
                "feature_extract": "Feature Extraction",
                "use_scheduler": "Use LR Scheduler",
                "train_progress": "Training Progress",
                "loss_curve": "Loss Curve",
                "accuracy_curve": "Accuracy Curve",
                "dataset_info": "Dataset Information",
                "model_info": "Model Information",
                "training_stats": "Training Statistics",
                "prediction_result": "Prediction Result",
                "confusion_matrix": "Confusion Matrix"
            }
        }

        self.setup_gui()
        self.create_widgets()
        self.update_language()  # 设置初始语言
        self.check_message_queue()  # 启动消息队列检查

    def setup_gui(self):
        """
        设置主GUI布局

        配置网格布局，确保窗口在调整大小时各组件能够正确伸缩
        """
        # 创建主框架，带有现代化样式
        self.main_frame = ttk.Frame(self.root, padding="20", style='Modern.TFrame')
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)

        # 创建现代化样式
        self.create_modern_styles()

    def create_modern_styles(self):
        """
        创建现代化的样式主题

        定义应用程序的视觉样式，包括颜色、字体、间距等
        """
        style = ttk.Style()

        # 设置整体主题色彩
        self.primary_color = '#2196F3'  # 主蓝色
        self.secondary_color = '#4CAF50'  # 成功绿色
        self.danger_color = '#f44336'  # 危险红色
        self.warning_color = '#FF9800'  # 警告橙色
        self.background_color = '#f8f9fa'  # 背景色
        self.surface_color = '#ffffff'  # 表面色
        self.text_color = '#333333'  # 文字色

        # 配置标签框架样式
        style.configure('Modern.TLabelframe', background=self.surface_color, relief='flat', borderwidth=1)
        style.configure('Modern.TLabelframe.Label', background=self.surface_color,
                       foreground=self.primary_color, font=('Arial', 12, 'bold'))

        # 配置按钮样式
        style.configure('Modern.TButton',
                       background=self.primary_color,
                       foreground='white',
                       font=('Arial', 10),
                       relief='flat',
                       padding=10)
        style.map('Modern.TButton',
                 background=[('active', '#1976D2'), ('pressed', '#1565C0')],
                 relief=[('pressed', 'sunken'), ('!pressed', 'flat')])

        # 配置成功按钮样式
        style.configure('Success.TButton',
                       background=self.secondary_color,
                       foreground='white',
                       font=('Arial', 10),
                       relief='flat',
                       padding=10)
        style.map('Success.TButton',
                 background=[('active', '#388E3C'), ('pressed', '#2E7D32')],
                 relief=[('pressed', 'sunken'), ('!pressed', 'flat')])

        # 配置危险按钮样式
        style.configure('Danger.TButton',
                       background=self.danger_color,
                       foreground='white',
                       font=('Arial', 10),
                       relief='flat',
                       padding=10)
        style.map('Danger.TButton',
                 background=[('active', '#d32f2f'), ('pressed', '#b71c1c')],
                 relief=[('pressed', 'sunken'), ('!pressed', 'flat')])

        # 配置标签样式
        style.configure('Modern.TLabel', background=self.surface_color, foreground=self.text_color, font=('Arial', 10))
        style.configure('Modern.TFrame', background=self.background_color)

        # 配置标题样式
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'),
                       foreground=self.primary_color, background=self.background_color)

        # 配置状态栏样式
        style.configure('Status.TLabel', font=('Arial', 9),
                       foreground='#666666', background=self.background_color)

    def create_widgets(self):
        """
        创建所有GUI组件

        构建用户界面的各个组件：
        - 标题区域（包含语言切换按钮）
        - 标签页（数据、模型、训练、测试）
        - 状态栏
        """
        # 标题和语言切换按钮行
        title_frame = ttk.Frame(self.main_frame, style='Modern.TFrame')
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        title_frame.columnconfigure(0, weight=1)

        title_label = ttk.Label(title_frame, text="ResNet50猫狗分类 - 迁移学习实验",
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, sticky=tk.W)

        # 语言切换按钮
        self.lang_button = ttk.Button(title_frame, text="中/EN", width=10,
                                    command=self.toggle_language, style='Modern.TButton')
        self.lang_button.grid(row=0, column=1, padx=(20, 0))

        # 创建标签页
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_frame.rowconfigure(1, weight=1)

        # 创建各个标签页
        self.create_data_tab()
        self.create_model_tab()
        self.create_training_tab()
        self.create_testing_tab()

        # 状态栏
        self.status_label = ttk.Label(self.main_frame, text="准备开始迁移学习实验。",
                                     style='Status.TLabel')
        self.status_label.grid(row=2, column=0, pady=(15, 0))

    def create_data_tab(self):
        """创建数据设置标签页"""
        self.data_frame = ttk.Frame(self.notebook, padding="20", style='Modern.TFrame')
        self.notebook.add(self.data_frame, text="数据设置")

        # 数据集选择框架
        dataset_frame = ttk.LabelFrame(self.data_frame, text="数据集设置",
                                     padding="15", style='Modern.TLabelframe')
        dataset_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        dataset_frame.columnconfigure(1, weight=1)

        # 数据集目录选择
        ttk.Label(dataset_frame, text="数据集目录:", style='Modern.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dataset_path_var = tk.StringVar()
        self.dataset_path_entry = ttk.Entry(dataset_frame, textvariable=self.dataset_path_var,
                                          font=('Arial', 10), width=50)
        self.dataset_path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 10), pady=5)

        self.select_dataset_button = ttk.Button(dataset_frame, text="选择目录",
                                              command=self.select_dataset, style='Modern.TButton')
        self.select_dataset_button.grid(row=0, column=2, pady=5)

        # 数据加载按钮
        self.load_data_button = ttk.Button(dataset_frame, text="加载数据",
                                         command=self.load_data, style='Success.TButton')
        self.load_data_button.grid(row=1, column=0, columnspan=3, pady=(10, 0))

        # 数据集信息显示
        info_frame = ttk.LabelFrame(self.data_frame, text="数据集信息",
                                   padding="15", style='Modern.TLabelframe')
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)

        # 创建数据显示文本框
        self.data_info_text = tk.Text(info_frame, height=15, width=80, wrap=tk.WORD,
                                     font=('Consolas', 10), bg='#f5f5f5')
        self.data_info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 滚动条
        data_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL,
                                     command=self.data_info_text.yview)
        data_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.data_info_text.configure(yscrollcommand=data_scrollbar.set)

        # 数据预览框架
        preview_frame = ttk.LabelFrame(self.data_frame, text="数据预览",
                                     padding="15", style='Modern.TLabelframe')
        preview_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(preview_frame, bg=self.surface_color,
                                      height=300, highlightthickness=1)
        self.preview_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置数据标签页的网格权重
        self.data_frame.columnconfigure(0, weight=1)
        self.data_frame.rowconfigure(2, weight=1)

    def create_model_tab(self):
        """创建模型配置标签页"""
        self.model_frame = ttk.Frame(self.notebook, padding="20", style='Modern.TFrame')
        self.notebook.add(self.model_frame, text="模型配置")

        # 模型设置框架
        model_config_frame = ttk.LabelFrame(self.model_frame, text="模型配置",
                                          padding="15", style='Modern.TLabelframe')
        model_config_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        model_config_frame.columnconfigure(1, weight=1)

        # 训练策略选择
        ttk.Label(model_config_frame, text="训练策略:", style='Modern.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.strategy_var = tk.StringVar(value="fine_tune")
        strategy_frame = ttk.Frame(model_config_frame)
        strategy_frame.grid(row=0, column=1, sticky=tk.W, pady=5)

        ttk.Radiobutton(strategy_frame, text="微调模式", variable=self.strategy_var,
                       value="fine_tune").grid(row=0, column=0, padx=(0, 20))
        ttk.Radiobutton(strategy_frame, text="特征提取", variable=self.strategy_var,
                       value="feature_extract").grid(row=0, column=1)

        # 训练参数设置
        param_frame = ttk.LabelFrame(self.model_frame, text="训练参数",
                                    padding="15", style='Modern.TLabelframe')
        param_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        param_frame.columnconfigure(1, weight=1)

        # 批次大小
        ttk.Label(param_frame, text="批次大小:", style='Modern.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.batch_size_var = tk.IntVar(value=16)
        ttk.Spinbox(param_frame, from_=1, to=64, textvariable=self.batch_size_var,
                   width=10).grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # 训练轮数
        ttk.Label(param_frame, text="训练轮数:", style='Modern.TLabel').grid(row=1, column=0, sticky=tk.W, pady=5)
        self.epochs_var = tk.IntVar(value=10)
        ttk.Spinbox(param_frame, from_=1, to=100, textvariable=self.epochs_var,
                   width=10).grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # 学习率
        ttk.Label(param_frame, text="学习率:", style='Modern.TLabel').grid(row=2, column=0, sticky=tk.W, pady=5)
        self.learning_rate_var = tk.DoubleVar(value=0.001)
        ttk.Entry(param_frame, textvariable=self.learning_rate_var,
                 width=10).grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # 学习率调度器
        self.use_scheduler_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="使用学习率调度器",
                       variable=self.use_scheduler_var).grid(row=3, column=0, columnspan=2,
                                                          sticky=tk.W, pady=10)

        # 模型构建按钮
        self.build_model_button = ttk.Button(param_frame, text="构建模型",
                                           command=self.build_model, style='Success.TButton')
        self.build_model_button.grid(row=4, column=0, columnspan=2, pady=(10, 0))

        # 模型信息显示
        info_frame = ttk.LabelFrame(self.model_frame, text="模型信息",
                                   padding="15", style='Modern.TLabelframe')
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)

        self.model_info_text = tk.Text(info_frame, height=15, width=80, wrap=tk.WORD,
                                     font=('Consolas', 10), bg='#f5f5f5')
        self.model_info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 滚动条
        model_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL,
                                      command=self.model_info_text.yview)
        model_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.model_info_text.configure(yscrollcommand=model_scrollbar.set)

        # 配置模型标签页的网格权重
        self.model_frame.columnconfigure(0, weight=1)
        self.model_frame.rowconfigure(2, weight=1)

    def create_training_tab(self):
        """创建训练监控标签页"""
        self.training_frame = ttk.Frame(self.notebook, padding="20", style='Modern.TFrame')
        self.notebook.add(self.training_frame, text="训练监控")

        # 训练控制框架
        control_frame = ttk.LabelFrame(self.training_frame, text="训练控制",
                                     padding="15", style='Modern.TLabelframe')
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        self.start_training_button = ttk.Button(control_frame, text="开始训练",
                                              command=self.start_training, style='Success.TButton')
        self.start_training_button.grid(row=0, column=0, padx=(0, 10))

        self.stop_training_button = ttk.Button(control_frame, text="停止训练",
                                             command=self.stop_training_func, style='Danger.TButton',
                                             state=tk.DISABLED)
        self.stop_training_button.grid(row=0, column=1)

        # 进度显示
        progress_frame = ttk.LabelFrame(self.training_frame, text="训练进度",
                                      padding="15", style='Modern.TLabelframe')
        progress_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        progress_frame.columnconfigure(0, weight=1)

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                          maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 进度标签
        self.progress_label = ttk.Label(progress_frame, text="准备开始训练...",
                                       style='Modern.TLabel')
        self.progress_label.grid(row=1, column=0, sticky=tk.W)

        # 训练统计框架
        stats_frame = ttk.LabelFrame(self.training_frame, text="训练统计",
                                    padding="15", style='Modern.TLabelframe')
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        # 统计信息显示
        self.stats_text = tk.Text(stats_frame, height=10, width=80, wrap=tk.WORD,
                                 font=('Consolas', 10), bg='#f5f5f5')
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # 图表显示框架
        charts_frame = ttk.LabelFrame(self.training_frame, text="训练图表",
                                     padding="15", style='Modern.TLabelframe')
        charts_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        charts_frame.columnconfigure(0, weight=1)
        charts_frame.rowconfigure(0, weight=1)

        self.chart_canvas = tk.Canvas(charts_frame, bg=self.surface_color,
                                    height=400, highlightthickness=1)
        self.chart_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置训练标签页的网格权重
        self.training_frame.columnconfigure(0, weight=1)
        self.training_frame.rowconfigure(3, weight=1)

    def create_testing_tab(self):
        """创建模型测试标签页"""
        self.testing_frame = ttk.Frame(self.notebook, padding="20", style='Modern.TFrame')
        self.notebook.add(self.testing_frame, text="模型测试")

        # 单张图片测试框架
        single_test_frame = ttk.LabelFrame(self.testing_frame, text="单张图片测试",
                                         padding="15", style='Modern.TLabelframe')
        single_test_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        single_test_frame.columnconfigure(1, weight=1)

        # 图片选择
        ttk.Label(single_test_frame, text="选择图片:", style='Modern.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.test_image_path_var = tk.StringVar()
        self.test_image_entry = ttk.Entry(single_test_frame, textvariable=self.test_image_path_var,
                                        font=('Arial', 10), width=50)
        self.test_image_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 10), pady=5)

        self.select_test_image_button = ttk.Button(single_test_frame, text="选择图片",
                                                 command=self.select_test_image, style='Modern.TButton')
        self.select_test_image_button.grid(row=0, column=2, pady=5)

        # 预测按钮
        self.predict_button = ttk.Button(single_test_frame, text="预测",
                                       command=self.predict_single_image, style='Success.TButton')
        self.predict_button.grid(row=1, column=0, columnspan=3, pady=(10, 0))

        # 预测结果显示
        result_frame = ttk.LabelFrame(self.testing_frame, text="预测结果",
                                     padding="15", style='Modern.TLabelframe')
        result_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        result_frame.columnconfigure(0, weight=1)

        # 图像显示
        self.test_image_canvas = tk.Canvas(result_frame, bg=self.surface_color,
                                         width=400, height=300, highlightthickness=1)
        self.test_image_canvas.grid(row=0, column=0, pady=(0, 10))

        # 结果文本
        self.prediction_result_label = ttk.Label(result_frame, text="请选择图片进行预测",
                                               style='Modern.TLabel')
        self.prediction_result_label.grid(row=1, column=0)

        # 批量测试框架
        batch_test_frame = ttk.LabelFrame(self.testing_frame, text="批量测试",
                                        padding="15", style='Modern.TLabelframe')
        batch_test_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        batch_test_frame.columnconfigure(0, weight=1)
        batch_test_frame.rowconfigure(1, weight=1)

        # 批量测试按钮
        self.batch_test_button = ttk.Button(batch_test_frame, text="在验证集上测试",
                                          command=self.batch_test, style='Modern.TButton')
        self.batch_test_button.grid(row=0, column=0, pady=(0, 10))

        # 批量测试结果显示
        self.batch_result_text = tk.Text(batch_test_frame, height=15, width=80, wrap=tk.WORD,
                                        font=('Consolas', 10), bg='#f5f5f5')
        self.batch_result_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置测试标签页的网格权重
        self.testing_frame.columnconfigure(0, weight=1)
        self.testing_frame.rowconfigure(2, weight=1)

    def toggle_language(self):
        """切换界面语言"""
        self.current_language = "en" if self.current_language == "zh" else "zh"
        self.update_language()

    def update_language(self):
        """根据当前语言更新所有UI文本"""
        texts = self.text_resources[self.current_language]

        # 更新窗口标题
        self.root.title(texts["app_title"])

        # 更新标签页标题
        self.notebook.tab(0, text=texts["data_tab"])
        self.notebook.tab(1, text=texts["model_tab"])
        self.notebook.tab(2, text=texts["training_tab"])
        self.notebook.tab(3, text=texts["testing_tab"])

        # 更新按钮文本
        if hasattr(self, 'select_dataset_button'):
            self.select_dataset_button.config(text=texts["select_dataset"])
            self.load_data_button.config(text=texts["load_data"])
            self.build_model_button.config(text=texts["build_model"])
            self.start_training_button.config(text=texts["start_training"])
            self.stop_training_button.config(text=texts["stop_training"])
            self.predict_button.config(text=texts["predict"])
            self.select_test_image_button.config(text=texts["select_image"])

        # 更新状态栏
        if not hasattr(self, 'image_datasets') or not self.image_datasets:
            self.status_label.config(text=texts["status_ready"])

    def select_dataset(self):
        """选择数据集目录"""
        directory = filedialog.askdirectory(title="选择数据集目录")
        if directory:
            self.dataset_path_var.set(directory)
            self.data_dir = directory

    def load_data(self):
        """加载数据集"""
        if not self.data_dir:
            messagebox.showwarning("警告", "请先选择数据集目录！")
            return

        try:
            self.update_status("正在加载数据集...")

            # 在后台线程中加载数据
            loading_thread = threading.Thread(target=self._load_data_thread)
            loading_thread.daemon = True
            loading_thread.start()

        except Exception as e:
            messagebox.showerror("错误", f"数据加载失败: {str(e)}")

    def _load_data_thread(self):
        """后台数据加载线程"""
        try:
            # 设置数据变换
            data_transform = setup_data_transforms()

            # 加载数据集
            self.image_datasets = load_datasets(self.data_dir, data_transform)

            # 创建数据加载器
            batch_size = self.batch_size_var.get()
            self.dataloaders = create_dataloaders(self.image_datasets, batch_size)

            # 更新UI
            self.message_queue.put(("data_loaded", None))

        except Exception as e:
            self.message_queue.put(("error", f"数据加载失败: {str(e)}"))

    def build_model(self):
        """构建模型"""
        if not self.image_datasets:
            messagebox.showwarning("警告", "请先加载数据集！")
            return

        try:
            self.update_status("正在构建模型...")

            # 在后台线程中构建模型
            building_thread = threading.Thread(target=self._build_model_thread)
            building_thread.daemon = True
            building_thread.start()

        except Exception as e:
            messagebox.showerror("错误", f"模型构建失败: {str(e)}")

    def _build_model_thread(self):
        """后台模型构建线程"""
        try:
            # 加载预训练模型
            model = load_pretrained_resnet50()

            # 修改模型进行迁移学习
            fine_tune = self.strategy_var.get() == "fine_tune"
            model = modify_model_for_transfer_learning(model, num_classes=2, fine_tune=fine_tune)

            # 移动到设备
            model = model.to(self.device)

            self.model = model

            # 更新UI
            self.message_queue.put(("model_built", None))

        except Exception as e:
            self.message_queue.put(("error", f"模型构建失败: {str(e)}"))

    def start_training(self):
        """开始训练"""
        if self.model is None:
            messagebox.showwarning("警告", "请先构建模型！")
            return

        if self.is_training:
            messagebox.showwarning("警告", "模型正在训练中！")
            return

        try:
            self.is_training = True
            self.stop_training = False
            self.training_history = []
            self.current_epoch = 0
            self.total_epochs = self.epochs_var.get()

            # 更新UI状态
            self.start_training_button.config(state=tk.DISABLED)
            self.stop_training_button.config(state=tk.NORMAL)
            self.notebook.tab(2, state='normal')  # 启用训练标签页

            # 在后台线程中训练模型
            self.training_thread = threading.Thread(target=self._training_thread)
            self.training_thread.daemon = True
            self.training_thread.start()

            # 切换到训练标签页
            self.notebook.select(2)

        except Exception as e:
            messagebox.showerror("错误", f"训练启动失败: {str(e)}")
            self.is_training = False

    def _training_thread(self):
        """后台训练线程"""
        try:
            # 定义损失函数和优化器
            criterion = torch.nn.CrossEntropyLoss()

            # 获取学习率
            learning_rate = self.learning_rate_var.get()

            # 根据策略设置优化器
            fine_tune = self.strategy_var.get() == "fine_tune"
            if fine_tune:
                optimizer = torch.optim.Adam([
                    {'params': self.model.fc.parameters(), 'lr': learning_rate},
                    {'params': self.model.layer4.parameters(), 'lr': learning_rate * 0.1}
                ], lr=learning_rate * 0.1)
            else:
                optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=learning_rate)

            # 学习率调度器
            use_scheduler = self.use_scheduler_var.get()
            if use_scheduler:
                scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                           patience=2, verbose=False)
            else:
                scheduler = None

            # 创建模型保存目录
            os.makedirs('transferResNet50', exist_ok=True)
            model_path = 'transferResNet50/model_resnet50_cats_dogs.pth'

            # 训练循环
            best_model_wts = self.model.state_dict()
            best_acc = 0.0

            for epoch in range(self.total_epochs):
                if self.stop_training:
                    break

                self.current_epoch = epoch + 1

                # 训练和验证阶段
                for phase in ['train', 'valid']:
                    if self.stop_training:
                        break

                    if phase == 'train':
                        self.model.train()
                    else:
                        self.model.eval()

                    running_loss = 0.0
                    running_corrects = 0

                    for inputs, labels in self.dataloaders[phase]:
                        if self.stop_training:
                            break

                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                    epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)

                    # 记录训练历史
                    self.training_history.append({
                        'epoch': epoch + 1,
                        'phase': phase,
                        'loss': epoch_loss,
                        'accuracy': epoch_acc.item()
                    })

                    # 更新学习率
                    if phase == 'valid' and scheduler:
                        scheduler.step(epoch_acc)

                    # 保存最佳模型
                    if phase == 'valid' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = self.model.state_dict()
                        self.best_accuracy = best_acc.item()

                # 更新进度
                progress = ((epoch + 1) / self.total_epochs) * 100
                self.message_queue.put(("training_progress", progress))

                # 更新训练统计
                train_history = [h for h in self.training_history if h['phase'] == 'train']
                valid_history = [h for h in self.training_history if h['phase'] == 'valid']

                if train_history and valid_history:
                    latest_train = train_history[-1]
                    latest_valid = valid_history[-1]

                    stats_text = f"Epoch {epoch + 1}/{self.total_epochs}\n"
                    stats_text += f"Train Loss: {latest_train['loss']:.4f}, Train Acc: {latest_train['accuracy']:.4f}\n"
                    stats_text += f"Valid Loss: {latest_valid['loss']:.4f}, Valid Acc: {latest_valid['accuracy']:.4f}\n"
                    stats_text += f"Best Acc: {self.best_accuracy:.4f}\n"

                    self.message_queue.put(("training_stats", stats_text))

            # 训练完成
            if not self.stop_training:
                # 保存最佳模型
                torch.save(best_model_wts, model_path)
                self.message_queue.put(("training_complete", None))
            else:
                self.message_queue.put(("training_stopped", None))

        except Exception as e:
            self.message_queue.put(("error", f"训练过程出错: {str(e)}"))
        finally:
            self.is_training = False

    def stop_training_func(self):
        """停止训练"""
        if self.is_training:
            self.stop_training = True
            self.message_queue.put(("training_stopping", None))

    def select_test_image(self):
        """选择测试图片"""
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.test_image_path_var.set(file_path)
            self.display_test_image(file_path)

    def display_test_image(self, image_path):
        """显示测试图片"""
        try:
            # 加载并显示图片
            image = Image.open(image_path)

            # 调整图片大小
            image.thumbnail((350, 250), Image.Resampling.LANCZOS)

            # 转换为Tkinter格式
            photo = ImageTk.PhotoImage(image)

            # 显示在画布上
            self.test_image_canvas.delete("all")
            x = (400 - photo.width()) // 2
            y = (300 - photo.height()) // 2
            self.test_image_canvas.create_image(x, y, anchor=tk.NW, image=photo)

            # 保存引用防止被垃圾回收
            self.test_image_photo = photo

        except Exception as e:
            messagebox.showerror("错误", f"图片加载失败: {str(e)}")

    def predict_single_image(self):
        """预测单张图片"""
        if self.model is None:
            messagebox.showwarning("警告", "请先构建并训练模型！")
            return

        image_path = self.test_image_path_var.get()
        if not image_path or not os.path.exists(image_path):
            messagebox.showwarning("警告", "请选择有效的图片文件！")
            return

        try:
            # 在后台线程中进行预测
            prediction_thread = threading.Thread(target=self._predict_single_thread, args=(image_path,))
            prediction_thread.daemon = True
            prediction_thread.start()

        except Exception as e:
            messagebox.showerror("错误", f"预测失败: {str(e)}")

    def _predict_single_thread(self, image_path):
        """后台单张图片预测线程"""
        try:
            # 设置模型为评估模式
            self.model.eval()

            # 加载和预处理图片
            image = Image.open(image_path)

            # 数据预处理
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            input_tensor = transform(image).unsqueeze(0).to(self.device)

            # 预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                _, predicted = torch.max(outputs, 1)

            # 获取类别名称
            class_names = ['cat', 'dog']
            predicted_class = class_names[predicted.item()]
            confidence = probabilities[predicted.item()].item()

            # 发送结果到UI
            result_text = f"预测结果: {predicted_class}\n置信度: {confidence:.4f}"
            self.message_queue.put(("prediction_result", result_text))

        except Exception as e:
            self.message_queue.put(("error", f"预测失败: {str(e)}"))

    def batch_test(self):
        """批量测试"""
        if self.model is None:
            messagebox.showwarning("警告", "请先构建并训练模型！")
            return

        if not self.dataloaders:
            messagebox.showwarning("警告", "请先加载数据集！")
            return

        try:
            # 在后台线程中进行批量测试
            batch_test_thread = threading.Thread(target=self._batch_test_thread)
            batch_test_thread.daemon = True
            batch_test_thread.start()

        except Exception as e:
            messagebox.showerror("错误", f"批量测试失败: {str(e)}")

    def _batch_test_thread(self):
        """后台批量测试线程"""
        try:
            self.model.eval()

            test_loader = self.dataloaders['valid']
            class_names = test_loader.dataset.classes

            correct = 0
            total = 0
            class_correct = [0] * len(class_names)
            class_total = [0] * len(class_names)

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # 统计每个类别的准确率
                    for i in range(labels.size(0)):
                        label = labels[i]
                        class_correct[label] += (predicted[i] == label).item()
                        class_total[label] += 1

            # 计算总体准确率
            overall_accuracy = correct / total

            # 计算每个类别的准确率
            class_accuracies = []
            for i in range(len(class_names)):
                if class_total[i] > 0:
                    acc = class_correct[i] / class_total[i]
                    class_accuracies.append(f"{class_names[i]}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")
                else:
                    class_accuracies.append(f"{class_names[i]}: N/A")

            # 发送结果到UI
            result_text = f"批量测试结果\n"
            result_text += f"总准确率: {overall_accuracy:.4f} ({correct}/{total})\n"
            result_text += f"各类别准确率:\n"
            for acc in class_accuracies:
                result_text += f"  {acc}\n"

            self.message_queue.put(("batch_test_result", result_text))

        except Exception as e:
            self.message_queue.put(("error", f"批量测试失败: {str(e)}"))

    def check_message_queue(self):
        """检查消息队列并更新UI"""
        try:
            while True:
                try:
                    message_type, message_data = self.message_queue.get_nowait()
                    self.process_message(message_type, message_data)
                except queue.Empty:
                    break
        except Exception as e:
            print(f"处理消息队列错误: {e}")

        # 每100毫秒检查一次
        self.root.after(100, self.check_message_queue)

    def process_message(self, message_type, message_data):
        """处理来自后台线程的消息"""
        if message_type == "error":
            messagebox.showerror("错误", message_data)

        elif message_type == "data_loaded":
            self.update_status(self.text_resources[self.current_language]["status_data_loaded"])
            self.display_dataset_info()

        elif message_type == "model_built":
            self.update_status(self.text_resources[self.current_language]["status_model_built"])
            self.display_model_info()

        elif message_type == "training_progress":
            self.progress_var.set(message_data)
            self.progress_label.config(text=f"训练进度: {message_data:.1f}%")

        elif message_type == "training_stats":
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, message_data)

        elif message_type == "training_complete":
            self.update_status(self.text_resources[self.current_language]["status_trained"])
            self.start_training_button.config(state=tk.NORMAL)
            self.stop_training_button.config(state=tk.DISABLED)
            messagebox.showinfo("成功", f"模型训练完成！最佳准确率: {self.best_accuracy:.4f}")

        elif message_type == "training_stopped":
            self.update_status("训练已停止")
            self.start_training_button.config(state=tk.NORMAL)
            self.stop_training_button.config(state=tk.DISABLED)

        elif message_type == "training_stopping":
            self.update_status("正在停止训练...")

        elif message_type == "prediction_result":
            self.prediction_result_label.config(text=message_data)

        elif message_type == "batch_test_result":
            self.batch_result_text.delete(1.0, tk.END)
            self.batch_result_text.insert(tk.END, message_data)

    def display_dataset_info(self):
        """显示数据集信息"""
        if not self.image_datasets:
            return

        info_text = "数据集信息\n"
        info_text += "=" * 50 + "\n\n"

        for split in ['train', 'valid']:
            if split in self.image_datasets:
                dataset = self.image_datasets[split]
                info_text += f"{split.upper()} 数据集:\n"
                info_text += f"  样本数量: {len(dataset)}\n"
                info_text += f"  类别: {dataset.classes}\n"
                info_text += f"  类别映射: {dataset.class_to_idx}\n"
                info_text += f"  变换操作: {dataset.transform}\n\n"

        self.data_info_text.delete(1.0, tk.END)
        self.data_info_text.insert(tk.END, info_text)

        # 显示数据预览
        self.display_data_preview()

    def display_data_preview(self):
        """显示数据预览"""
        if not self.dataloaders:
            return

        try:
            # 获取一个批次的数据
            train_loader = self.dataloaders['train']
            inputs, labels = next(iter(train_loader))

            # 显示前8张图片
            num_images = min(8, inputs.size(0))

            # 创建图像网格
            from torchvision.utils import make_grid
            grid = make_grid(inputs[:num_images], nrow=4, normalize=True)

            # 转换为numpy数组
            npimg = grid.numpy().transpose((1, 2, 0))

            # 转换为PIL图像
            pil_image = Image.fromarray((npimg * 255).astype(np.uint8))

            # 调整大小
            pil_image = pil_image.resize((600, 400), Image.Resampling.LANCZOS)

            # 转换为Tkinter格式
            photo = ImageTk.PhotoImage(pil_image)

            # 显示在画布上
            self.preview_canvas.delete("all")
            x = (self.preview_canvas.winfo_width() - photo.width()) // 2
            y = (self.preview_canvas.winfo_height() - photo.height()) // 2
            if x < 0: x = 0
            if y < 0: y = 0

            self.preview_canvas.create_image(x, y, anchor=tk.NW, image=photo)
            self.preview_photo = photo

            # 添加标签信息
            class_names = train_loader.dataset.classes
            labels_text = [class_names[labels[i].item()] for i in range(num_images)]
            self.preview_canvas.create_text(300, 20, text=f"样本标签: {labels_text}",
                                          font=("Arial", 12), anchor=tk.N)

        except Exception as e:
            print(f"数据预览错误: {e}")

    def display_model_info(self):
        """显示模型信息"""
        if not self.model:
            return

        info_text = "模型信息\n"
        info_text += "=" * 50 + "\n\n"
        info_text += f"模型类型: ResNet50 (迁移学习)\n"
        info_text += f"训练策略: {self.strategy_var.get()}\n"
        info_text += f"计算设备: {self.device}\n"
        info_text += f"参数数量: {sum(p.numel() for p in self.model.parameters()):,}\n"
        info_text += f"可训练参数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}\n\n"

        info_text += "模型结构:\n"
        info_text += str(self.model)

        self.model_info_text.delete(1.0, tk.END)
        self.model_info_text.insert(tk.END, info_text)

    def update_status(self, message):
        """更新状态栏"""
        self.status_label.config(text=message)

    def run(self):
        """启动应用程序"""
        self.root.mainloop()

# 保留原有的函数，供GUI调用
def setup_data_transforms():
    """设置数据预处理变换"""
    data_transform = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomRotation(15),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return data_transform

def load_datasets(data_dir, data_transform):
    """加载猫狗数据集"""
    image_datasets = {
        x: datasets.ImageFolder(
            root=os.path.join(data_dir, x),
            transform=data_transform[x]
        )
        for x in ["train", "valid"]
    }
    return image_datasets

def create_dataloaders(image_datasets, batch_size=16):
    """创建数据加载器"""
    dataloader = {
        x: torch.utils.data.DataLoader(
            dataset=image_datasets[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        for x in ["train", "valid"]
    }
    return dataloader

def load_pretrained_resnet50():
    """加载预训练的ResNet50模型"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    return model

def modify_model_for_transfer_learning(model, num_classes=2, fine_tune=False):
    """修改模型以适应迁移学习"""
    if fine_tune:
        for param in model.layer4.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)

    return model

# 主程序入口
if __name__ == "__main__":
    try:
        # 创建GUI应用程序实例
        app = ResNet50CatDogGUI()
        # 启动应用程序
        app.run()
    except Exception as e:
        print(f"程序启动失败: {e}")
        import traceback
        traceback.print_exc()