# 数字图像处理 (Digital Image Processing) 代码仓库

本仓库包含了**数字图像处理**课程的实验（Experiments）与项目（Projects）的完整实现代码。

## 仓库结构

仓库主要由两部分组成：
- `Experiments/`：基础图像处理实验
- `Projects/`：综合性图像处理项目

---

## 实验部分 (Experiments)

包含多个独立模块的图像处理基础实验验证与算法实现，复用函数集成在utils.py中。实验内容：

* **Exp4 - 频域与空间域滤波 (`Experiments/Exp4/main_4.py`)**
    * 二维傅里叶变换及其频谱可视化
    * 频域高斯低通滤波
    * 空间域拉普拉斯锐化
    * 频域高通滤波与高频强调滤波（结合直方图均衡化）
* **Exp5 - 噪声与图像复原 (`Experiments/Exp5/main_5.py`)**
    * 高斯、椒盐及乘性噪声生成与分析
    * 使用最大值/最小值/中值滤波器进行空间滤波去噪
    * 周期噪声生成及其频域表现
    * 运动模糊图像模拟及复原（逆滤波与维纳滤波）
    * 空间几何仿射变换
* **Exp6 - 彩色图像处理 (`Experiments/Exp6/main_6.py`)**
    * 彩色图像索引表达与抖动（Floyd-Steinberg算法）
    * 色彩空间转换（RGB, HSV, NTSC/YIQ）
    * 彩色图像平滑（HSV空间V通道处理）与拉普拉斯锐化
    * 彩色图像边缘检测（彩色梯度VG与分量梯度PPG）
    * 基于颜色相似度的图像分割（欧氏距离与马氏距离）
* **Exp7 - 小波变换与应用 (`Experiments/Exp7/main_7.py`)**
    * 快速小波变换 (Fast Wavelet Transform)
    * 基于小波系数的边缘检测与显示
    * 基于小波的图像平滑（高频分量置零）
    * 小波渐进式重构（Progressive Reconstruction）
* **Exp8 - 图像压缩 (`Experiments/Exp8/main_8.py`)**
    * 图像 Shannon 熵计算与 Hoffmann 理论压缩比计算
    * 像素间冗余与一维线性预测编码 (LPC)
    * 心理视觉冗余与改进的灰度量化 (IGS)
    * JPEG 有损压缩模拟与 RMSE 质量评估
* **Exp9 - 形态学图像处理 (`Experiments/Exp9/main_9.py`)**
    * 二值图像的膨胀与腐蚀
    * 开闭运算（去噪、指纹修复、提取轮廓）
    * 击中或击不中变换 (Hit-or-Miss Transform)
    * 图像细化与骨骼化提取拓扑中轴
    * 连通分量标记与伪彩色映射
    * 灰度图像形态学（顶帽变换 Top-Hat 解决光照不均）

---

## 项目部分 (Projects)

包含5个综合性图像处理大项目：

### 1. 混合图像与图像增强 (`Projects/1_hybrid/`)
基于 SIGGRAPH 2006 论文，通过结合一张图像的高频分量和另一张图像的低频分量，生成随观看距离改变内容的混合图像 (Hybrid Images)。包含手动对齐、高斯滤波以及傅里叶频谱分析。此外还包含对比度增强和色彩转换。

### 2. 图像纹理拼接与迁移 (`Projects/2_quilting/`)
实现 Efros & Freeman 的经典图像缝合与纹理生成算法：
* **Random Quilt**: 随机块拼接
* **Simple Quilt**: 基于 SSD (平方差和) 的重叠块匹配
* **Cut Quilt (Seam Finding)**: 最小代价接缝寻找消除拼接伪影
* **Texture Transfer**: 结合目标引导图的纹理迁移（如生成吐司人脸）

### 3. 梯度域图像融合 (`Projects/3_gradient-domain_fusion/`)
通过求解泊松方程（Poisson Equation）实现无缝图像融合：
* **Toy Problem**: 构建稀疏矩阵通过梯度重建图像
* **Poisson Blending**: 泊松融合，保持源图像梯度，无缝融入背景
* **Mixed Gradients Blending**: 混合梯度融合，保留前景与背景中最强的纹理信息
* 结合 GrabCut 的自动轮廓提取融合

### 4. 高动态范围 (HDR) 成像与光照渲染 (`Projects/4_lighting/`)
* 从多曝光的LDR序列中还原HDR辐射图：实现朴素平均、加权平均，以及基于 Debevec 方法的**相机响应函数 (CRF) 估计**。
* **全景映射 (Panoramic Transformation)**: 将镜像球反射转换为 Equirectangular 投影。
* 基于图像的照明 (Image-Based Lighting, IBL): 通过生成的HDR全景图，渲染并将3D虚拟物体合成至真实场景照片中。

### 5. 视频拼接与背景提取 (`Projects/5_video_stitching/`)
使用 SIFT 特征和 RANSAC 算法计算鲁棒的单应性矩阵 (Homography)：
* 两帧和多关键帧的全景图拼接
* 将所有视频帧映射到参考坐标系生成固定视角的映射视频
* 提取背景全景图（时间轴中值滤波）
* 结合背景消除生成纯前景（运动物体）视频与背景视频

---

## 安装与运行指南

### 1. 环境依赖
本项目使用 Python 编写，主要依赖项定义在 `requirements.txt` 中。
请在 Python 3.10+ 环境中安装依赖：

```bash
pip install -r requirements.txt
```

**依赖库列表：**

* `opencv-python`
* `numpy`
* `matplotlib`
* `scipy`
* `pillow`
* `pywavelets`

### 2. 数据准备

由于部分图像体积较大且属于个人/受版权保护内容，`.gitignore` 配置忽略了仓库中的 `data/`、`pictures/` 和 `samples/` 目录。
运行前，请确保在对应实验或项目的根目录下准备好相关测试图像资源，并根据各 `main` 脚本底部的 `if __name__ == '__main__':` 调用更改为您本地的图片路径。

### 3. 运行示例

直接通过 Python 运行各个实验的 `main.py` 脚本，例如：

```bash
python Experiments/Exp4/main_4.py
python Projects/2_quilting/main.py

```