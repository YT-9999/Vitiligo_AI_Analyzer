# Vitiligo_AI_Analyzer

白癜风智能辅助分析系统 —— 基于 OpenCV、深度学习（U-Net）和单目深度估计（Depth Pro）的全自动图像处理工具。

## ✨ 主要功能

本系统可对手机拍摄的白癜风患处照片进行全自动分析，输出：
- **白斑区域分割结果**（背景标红，白斑保留灰度纹理）
- **1mm 网格颜色方块矩阵图**（用于评估色素分布）
- **微针点胶机参数表**（CSV 格式，包含每个 1mm² 网格的 X/Y 坐标及建议微针高度）
- **白纸黑点定位图**（用于点胶机定位）

### 三种物理尺寸获取方式
1. **手动输入**：用户自行测量并输入照片的实际物理宽度和高度（毫米）。
2. **CSV 批量导入**：通过表格文件批量提供每张照片的尺寸。
3. **AI 3D 测量**（推荐）：基于 Apple Depth Pro 单目深度估计模型，全自动计算照片中皮肤区域的真实物理尺寸，无需放置参照物。

### 三种白斑分割模式
1. **手动框选**：用户手动框选白斑区域，程序自动计算最佳阈值。
2. **固定阈值**：手动输入 0-255 的全局二值化阈值。
3. **AI 自动分割**（推荐）：基于 U-Net 深度学习模型，全自动识别白斑区域，无需人工干预。

## 📁 项目结构
.
├── main.py # 主程序入口

├── utils_depth.py # Depth Pro 工具函数

├── requirements.txt # Python 依赖列表

├── LICENSE # MIT 许可证

├── README.md # 项目说明文档

├── checkpoints/ # Depth Pro 预训练权重（需自行下载）

└── vitiligo_model.onnx # U-Net 分割模型（需自行训练或下载）

## 🚀 快速开始

### 环境要求
- Windows 10/11
- Anaconda 或 Miniconda（推荐）
- NVIDIA GPU（可选，用于加速 AI 推理）

### 安装步骤
1. 克隆本仓库：

   ```bash
   git clone https://github.com/YT-9999/Vitiligo_AI_Analyzer.git
   cd Vitiligo_AI_Analyzer

2. 创建并激活 Conda 虚拟环境：

   ```bash
   conda create -n vitiligo_ai python=3.10 -y
   conda activate vitiligo_ai

3. 安装依赖：

   ```bash
   pip install -r requirements.txt

4. 下载 Depth Pro 预训练权重：

   访问 Apple ml-depth-pro 仓库，按照说明下载 depth_pro.pt，并将其放置于 checkpoints/ 目录下。

5. 准备 U-Net 分割模型：

   将你训练好的 ONNX 模型文件 vitiligo_model.onnx 及对应的外部数据文件 vitiligo_model.onnx.data 放置于项目根目录。

6. 运行程序：

   ```bash
   python main.py

### 使用打包好的可执行文件（无需 Python 环境）
1. 从 Releases 页面下载最新的 VitiligoAnalyzer.zip。

2. 解压到任意目录（路径不含中文）。

3. 双击 run.bat 即可运行。

## 📊 输出文件说明

每张处理后的图片都会在 output/ 目录下生成一个以原文件名命名的子文件夹，包含：

| 文件名 | 说明 |
|--------|------|
| *_vitiligo_result.png | 白斑保留灰度纹理、背景标红的中间结果图 |
| *_mosaic_result.png | 1mm 网格颜色方块矩阵图 |
| *_needle_params.csv | 点胶机参数表（列：序号, X, Y, Height_um, avg_B, avg_G, avg_R） |
| *_dot_map.png | 白纸黑点图，用于点胶机定位 |
| *_grid_color_data.csv | 所有网格的原始颜色数据 |

## ⚠️ 注意事项

- **拍摄规范**：为获得最佳 AI 3D 测量效果，请尽量在光线充足的环境下垂直拍摄患处。

- **磁盘空间**：程序运行时会在当前目录创建临时文件夹（若以run.bat运行），请确保磁盘有至少 2GB 可用空间。

- **杀毒软件**：本程序使用 PyArmor 代码混淆及 PyInstaller 打包，可能被部分杀毒软件误报。请添加信任或暂时关闭杀毒软件。

## 📜 许可证

本项目采用 MIT License。

版权所有 (c) 2026 汤涛绮。

## 🙏 致谢

- **Apple ml-depth-pro** —— 单目深度估计模型

- **ONNX Runtime** —— 跨平台推理引擎

- **PyTorch** —— 深度学习框架

- **OpenCV** —— 计算机视觉库

## 📧 联系方式

如有问题或建议，请联系原作者：（[2432622861@qq.com]）
