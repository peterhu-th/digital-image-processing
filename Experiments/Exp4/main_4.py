import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import show_images

def fourier_transform(img_path):
    """
    二维傅立叶变换：（空域->频域）
        1. 图像滤波与增强
            低通：滤除高频实现平滑/去噪；高通：锐化/边缘提取
        2. 周期性噪声去除
            把网纹、条纹转化成孤立亮点，可用陷波滤波去除
        3. 图像压缩
            利用能量集中特性，阶段高频分量，压缩数据量
    """
    f = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if f is None:
        print(f"找不到图片: {img_path}")
        return

    # 傅里叶变换
    F = np.fft.fft2(f)
    # 频谱移动到中心
    FC = np.fft.fftshift(F)
    # 对频谱作对数变换，拓展其动态范围
    S2 = np.log(1 + np.abs(FC))

    show_images([f, np.abs(F), S2],
                ['原图', '未居中的频谱 (能量在四角)', '居中且对数变换后的频谱'],
                1, 3, "二维傅立叶变换")


def gaussian_lowpass(img_path):
    """
    频域高斯低通滤波：（频域）
        传递函数平滑，不产生伪影
        适用于滤除高频噪声、重采样前的抗混滤波器、模糊处理
    """
    f = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if f is None:
        print(f"找不到图片: {img_path}")
        return

    M, N = f.shape
    # 扩充图像尺寸以满足FFT计算需要
    P, Q = 2 * M, 2 * N
    # 得到填充图像的傅里叶变换并居中
    F = np.fft.fft2(f, s=(P, Q))
    F_shift = np.fft.fftshift(F)
    # 生成频域坐标网格
    u = np.arange(-P // 2, P // 2)
    v = np.arange(-Q // 2, Q // 2)
    U, V = np.meshgrid(v, u)
    D_squared = U ** 2 + V ** 2
    # 截止频率
    D0 = 0.05 * Q
    # 生成高斯低通滤波器
    H = np.exp(-D_squared / (2 * (D0 ** 2)))
    # 将傅里叶变换的结果与滤波器函数相乘
    G_shift = F_shift * H
    # 逆变换回空间域，获得傅里叶变换的实部
    G = np.fft.ifftshift(G_shift)
    g_padded = np.fft.ifft2(G)
    g_padded = np.real(g_padded)
    # 将左上角的矩形修剪为原始大小
    g = g_padded[:M, :N]

    show_images([f, H, g],
                ['原图', '高斯低通滤波器 (频域)', '滤波结果'],
                1, 3, "频域高斯低通滤波")


def laplacian_sharpening(img_path):
    """
    拉普拉斯空间锐化滤波：（空域）
        算子具有旋转不变性，对图像中任何方向的边缘突变有相同相应
        放大孤立噪点，需要图像信噪比高
        适用显微成像、天文观测
    """
    a = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if a is None:
        print(f"找不到图片: {img_path}")
        return

    # 拉普拉斯算子模板
    h = np.array([[1, 1, 1],
                  [1, -8, 1],
                  [1, 1, 1]], dtype=np.float32)
    # 用拉普拉斯算子锐化滤波
    b = cv2.filter2D(a, cv2.CV_32F, h)
    # 锐化：原图叠加滤波结果
    K = cv2.subtract(a.astype(np.float32), b)
    # 将数值截断在0-255之间以便正确显示
    K = np.clip(K, 0, 255).astype(np.uint8)
    b_display = np.clip(np.abs(b), 0, 255).astype(np.uint8)

    show_images([a, b_display, K],
                ['原图', '提取出的拉普拉斯图像', '锐化增强后结果'],
                1, 3, "拉普拉斯空间域锐化")


def highpass_and_emphasis(img_path):
    """
    频域高通滤波器与高频强调滤波：（频域->空域）
        保留背景，提高对比度
        适用X射线增强
    """
    f = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if f is None:
        print(f"找不到图片: {img_path}")
        return

    M, N = f.shape
    P, Q = 2 * M, 2 * N
    F = np.fft.fft2(f, s=(P, Q))
    F_shift = np.fft.fftshift(F)
    # 坐标网格计算距离
    u = np.arange(-P // 2, P // 2)
    v = np.arange(-Q // 2, Q // 2)
    U, V = np.meshgrid(v, u)
    D = np.sqrt(U ** 2 + V ** 2)
    D[D == 0] = 1e-5  # 防止除零
    # 截止频率与2阶巴特沃斯高通滤波器
    D0 = 0.05 * P
    n = 2
    HBW = 1 / (1 + (D0 / D) ** (2 * n))
    # 高频强调滤波器
    H = 0.5 + 2 * HBW
    # 仅高通滤波
    G_BHPF_shift = F_shift * HBW
    gbw = np.real(np.fft.ifft2(np.fft.ifftshift(G_BHPF_shift)))[:M, :N]
    # 归一化
    gbw_norm = cv2.normalize(gbw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 高频强调滤波
    G_Emp_shift = F_shift * H
    ghf = np.real(np.fft.ifft2(np.fft.ifftshift(G_Emp_shift)))[:M, :N]
    ghf_norm = cv2.normalize(ghf, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 直方图均衡化
    ghe = cv2.equalizeHist(ghf_norm)

    show_images([f, gbw_norm, ghf_norm, ghe],
                ['原图 (X光)', '高通滤波 (偏暗)', '高频强调滤波 (保留低频+增强高频)', '高频强调 + 直方图均衡化'],
                1, 4, "4.3.5 高频强调滤波与直方图均衡化")


if __name__ == '__main__':
    fourier_transform(os.path.join('data', 'chestXray.tif'))
    gaussian_lowpass(os.path.join('data', 'points.tif'))
    laplacian_sharpening(os.path.join('data', 'moon.tif'))
    highpass_and_emphasis(os.path.join('data', 'chestXray.tif'))