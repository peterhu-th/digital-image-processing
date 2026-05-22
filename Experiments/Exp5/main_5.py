import numpy as np
import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import show_images, load_grayscale_image


def random_noise_effects(img_path):
    """
    随机噪声对图像的影响
    """
    f_uint8 = load_grayscale_image(img_path)
    if f_uint8 is None: return

    # 归一化
    f = f_uint8.astype(np.float32) / 255.0
    # 叠加高斯噪声
    noise_gaussian = np.random.normal(0, 0.1, f.shape)
    g1 = np.clip(f + noise_gaussian, 0, 1)
    # 叠加椒盐噪声
    g2 = f.copy()
    rand_mat = np.random.rand(*f.shape) # 生成与原图像形状相同的随机矩阵
    g2[rand_mat < 0.01] = 0.0  # 1% 为胡椒噪声
    g2[(rand_mat >= 0.01) & (rand_mat < 0.02)] = 1.0  # 1% 为盐粒噪声
    # 叠加乘性噪声
    noise_speckle = np.random.normal(0, 0.1, f.shape)
    g3 = np.clip(f + f * noise_speckle, 0, 1)

    show_images([f, g1, g2, g3],
                ['原始图像', '高斯噪声图像', '椒盐噪声图像', '乘性噪声图像'],
                2, 2, "随机噪声以及对图像的影响")


def spatial_filtering_noise_removal(img_path):
    """
    空间滤波去除随机噪声
    """
    f = load_grayscale_image(img_path)
    if f is None: return

    # 胡椒噪声
    gp = f.copy()
    gp[np.random.rand(*f.shape) < 0.1] = 0
    # 盐粒噪声
    gs = f.copy()
    gs[np.random.rand(*f.shape) < 0.1] = 255

    # 定义 3x3 结构元素
    kernel = np.ones((3, 3), np.uint8)
    # 使用最大值滤波器去除胡椒噪声
    fpmax = cv2.dilate(gp, kernel)
    # 使用最小值滤波器去除盐粒噪声
    fsmin = cv2.erode(gs, kernel)
    # 使用中值滤波器去除盐粒噪声
    fs = cv2.medianBlur(gs, 3)

    show_images([f, gp, gs, fpmax, fsmin, fs],
                ['原始图像', '仅叠加胡椒噪声', '仅叠加盐粒噪声',
                 '最大值滤波去胡椒', '最小值滤波去盐粒', '中值滤波去盐粒'],
                2, 3, "空间滤波去除随机噪声")


def periodic_noise(img_path):
    """
    周期噪声以及对图像的影响
    """
    f = load_grayscale_image(img_path)
    if f is None: return

    M, N = f.shape
    # 周期噪声的频域位置设定与振幅
    u0, v0 = 32, 6  # 水平方向 32 个周期，垂直方向 6 个周期
    A = 64
    # 生成包含坐标的一维数组
    x = np.arange(N)
    y = np.arange(M)
    X, Y = np.meshgrid(x, y)
    # 生成空间域二维正弦波作为周期噪声
    r = A * np.sin(2 * np.pi * (u0 * X / N + v0 * Y / M))
    # 噪声图像叠加测试图像
    g = np.clip(f + r, 0, 255).astype(np.uint8)
    # 转换为频谱
    R = np.fft.fftshift(np.fft.fft2(r))
    S = np.log(1 + np.abs(R))
    show_images([f, S, r, g],
                ['原始图像', '周期噪声频谱 (特定冲击)', '周期噪声空间图像', '叠加退化图像'],
                2, 2, "周期噪声以及对图像的影响")


def motion_blur_and_restoration(img_path):
    """
    运动模糊图像建模与维纳滤波消除：
        逆滤波：F_bar = G/H 高频部分的 H 很小，噪声过大
        维纳滤波：F_bar = F + N/H 克服了逆滤波的噪声放大缺陷
        核心思想：使复原图像和原始图像均方误差最小
    """
    f_uint8 = load_grayscale_image(img_path)
    if f_uint8 is None: return
    f_float = f_uint8.astype(np.float32) / 255.0

    # 创建运动模糊退化函数 PSF (模拟 45 度角)
    size = 7
    psf = np.zeros((size, size), dtype=np.float32)
    np.fill_diagonal(psf, 1)
    psf /= psf.sum()
    # 产生退化图像并叠加高斯噪声
    gb = cv2.filter2D(f_float, -1, psf)
    noise = np.random.normal(0, np.sqrt(0.001), gb.shape)
    g = np.clip(gb + noise, 0, 1)
    G = np.fft.fft2(g)

    # 填充 PSF 到图像大小并居中对齐
    psf_padded = np.zeros_like(g)
    psf_padded[:size, :size] = psf
    psf_padded = np.roll(psf_padded, -size // 2, axis=0)
    psf_padded = np.roll(psf_padded, -size // 2, axis=1)
    H = np.fft.fft2(psf_padded)
    H_conj = np.conj(H)
    H_mag_sq = np.abs(H) ** 2
    # 逆滤波
    F_inv = G * H_conj / (H_mag_sq + 1e-6)
    fr1 = np.clip(np.real(np.fft.ifft2(F_inv)), 0, 1)
    # 维纳滤波 (噪信比 R)
    R = 0.01
    F_wiener = G * H_conj / (H_mag_sq + R)
    fr2 = np.clip(np.real(np.fft.ifft2(F_wiener)), 0, 1)

    show_images([f_uint8, g, fr1, fr2],
                ['原始图像', '加噪和运动模糊图像', '直接逆滤波复原', '常数比率维纳滤波复原'],
                2, 2, "运动模糊与复原")


def affine_transform(img_path):
    """
    仿射变换：
        拉伸/压缩、平移、旋转、错切
        保持共线和平行关系，可用 2 × 3 矩阵表示
    """
    f = load_grayscale_image(img_path)
    if f is None: return

    s = 0.8  # 缩放系数
    theta = 30  # 旋转角度
    h, w = f.shape
    cx, cy = w // 2, h // 2  # 图像中心作为旋转中心
    # 获取仿射变换矩阵 T
    T = cv2.getRotationMatrix2D((cx, cy), angle=theta, scale=s)
    # 对图像施加仿射变换
    g = cv2.warpAffine(f, T, (w, h))

    show_images([f, g],
                ['原始图像', '仿射变换图像 (缩放0.8, 旋转30度)'],
                1, 2, "空间几何仿射变换")


def frequency_filtering(img_path):
    """
    频域滤波
    """
    f_uint8 = load_grayscale_image(img_path)
    if f_uint8 is None: return

    f = f_uint8.astype(np.float32) / 255.0

    # 产生运动模糊与高斯噪声
    psf = np.zeros((7, 7), dtype=np.float32)
    np.fill_diagonal(psf, 1)
    psf /= psf.sum()
    gb = cv2.filter2D(f, -1, psf)
    g = np.clip(gb + np.random.normal(0, np.sqrt(0.01), gb.shape), 0, 1)

    # 获取填充尺寸和中心化频谱
    M, N = g.shape
    P, Q = 2 * M, 2 * N
    F_shift = np.fft.fftshift(np.fft.fft2(g, s=(P, Q)))

    # 生成频域坐标网格
    u = np.arange(-P // 2, P // 2)
    v = np.arange(-Q // 2, Q // 2)
    U, V = np.meshgrid(v, u)
    D = np.sqrt(U ** 2 + V ** 2)
    D[D == 0] = 1e-5

    # 高斯低通滤波器
    D0_low = 0.05 * Q
    H_low = np.exp(-(D ** 2) / (2 * (D0_low ** 2)))
    g_low = np.real(np.fft.ifft2(np.fft.ifftshift(F_shift * H_low)))[:M, :N]

    # 二阶巴特沃斯高通滤波器
    D0_high = 0.05 * P
    HBW = 1 / (1 + (D0_high / D) ** 4)
    g_high = np.real(np.fft.ifft2(np.fft.ifftshift(F_shift * HBW)))[:M, :N]

    # 高频强调滤波器
    H_emp = 0.5 + 2 * HBW
    g_emp = np.real(np.fft.ifft2(np.fft.ifftshift(F_shift * H_emp)))[:M, :N]

    show_images([g, g_low, g_high, g_emp],
                ['加噪退化图像 (模糊+噪声)', '高斯低通滤波',
                 '巴特沃斯高通滤波', '高频强调滤波'],
                2, 2, "退化图像的频域滤波对比")


if __name__ == '__main__':
    random_noise_effects(os.path.join('data', 'pattern.tif'))
    spatial_filtering_noise_removal(os.path.join('data', 'trees.tif'))
    periodic_noise(os.path.join('data', 'kids.tif'))
    motion_blur_and_restoration(os.path.join("data", "pixeldup.tif"))
    affine_transform(os.path.join("data", "pixeldup.tif"))
    frequency_filtering(os.path.join("data", "pixeldup.tif"))