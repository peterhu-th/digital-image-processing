import numpy as np
import cv2
import os
import sys
import pywt
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import show_images, load_grayscale_image, build_dwt_image


def fast_wavelet_transform():
    """
    快速小波变换 (Mallet)：
        通过低通 + 高通滤波跳过内积计算，每次滤波后下采样，时间复杂度 O(N)
    """
    wavelet = pywt.Wavelet('haar')
    print("Lo_D (分解低通):", wavelet.dec_lo)
    print("Hi_D (分解高通):", wavelet.dec_hi)
    print("Lo_R (重构低通):", wavelet.rec_lo)
    print("Hi_R (重构高通):", wavelet.rec_hi)
    print(wavelet)
    # 魔方矩阵 f=magic(4)
    f = np.array([[16, 2, 3, 13],
                  [5, 11, 10, 8],
                  [9, 7, 6, 12],
                  [4, 14, 15, 1]], dtype=np.float64)
    # 2 层离散小波分解，返回 [cA2, (cH2, cV2, cD2), (cH1, cV1, cD1)]
    # cA2：第 2 层低频近似系数（模糊轮廓）
    # cH cV cD：水平 (Horizontal)、垂直 (Vertical)、对角线 (diagonal) 方向的高频细节系数
    coeffs = pywt.wavedec2(f, 'haar', level=2)
    print("\n=== magic(4) 的 2 层 Haar 小波分解近似系数 ===")
    print(coeffs[0])
    # 产生尺度向量和小波向量，层数越高曲线越平滑
    phi, psi, xval = wavelet.wavefun(level=10)
    # 生成与 xval 形状相同的数组，元素全为 0
    xaxis = np.zeros_like(xval)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(xval, psi, 'k', xval, xaxis, '-k')
    plt.axis((0, 1, -1.5, 1.5))
    plt.axis('square')
    plt.title('Haar Wavelet Function')

    plt.subplot(122)
    plt.plot(xval, phi, 'k', xval, xaxis, '-k')
    plt.axis((0, 1, -1.5, 1.5))
    plt.axis('square')
    plt.title('Haar Scaling Function')
    plt.tight_layout()
    plt.show()


def wavelet_coefficients_and_edges(img_path):
    """
    显示小波变换系数并提取边缘：
        拆分大致轮廓和边缘细节，舍弃轮廓保留细节
    """
    if not img_path: return
    f = load_grayscale_image(img_path)
    if f is None: return
    # 1 层小波分解，返回 (cA, (cH, cV, cD))
    coeffs = pywt.wavedec2(f, 'sym4', level=1)
    img_dwt = build_dwt_image(coeffs)
    # 把元组转换为列表以修改数据
    coeffs_cut = list(coeffs)
    # 修改低频近似系数 cA
    coeffs_cut[0] = np.zeros_like(coeffs_cut[0])
    # 生成与 cA 尺寸相同的纯黑矩阵
    img_dwt_cut = build_dwt_image(coeffs_cut)
    # 小波反变换并求绝对值
    edges = np.abs(pywt.waverec2(coeffs_cut, 'sym4'))
    edges_norm = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    show_images([f, img_dwt, img_dwt_cut, edges_norm],
                ['原图', '小波分解系数', '去掉近似系数后的系数', '反变换后的边缘图像'],
                2, 2, "小波变换系数显示与边缘检测")


def wavelet_image_smoothing(img_path):
    """
    基于小波的图像平滑：
        变换层数越浅，包含细节越微小；
        最低频近似层包含图像绝大部分能量（基础亮度和大面积色彩）
    """
    if not img_path: return
    f = load_grayscale_image(img_path)
    if f is None: return

    # 4 层小波分解
    coeffs = pywt.wavedec2(f, 'sym4', level=4)
    img_dwt = build_dwt_image(coeffs)
    recon_images = [f]
    coeffs_mod = list(coeffs)
    # 清空特定层级高频系数
    coeffs_mod[4] = tuple(np.zeros_like(x) for x in coeffs_mod[4])
    # 用修改后的系数重构图像
    recon_images.append(np.clip(pywt.waverec2(coeffs_mod, 'sym4'), 0, 255).astype(np.uint8))
    coeffs_mod[3] = tuple(np.zeros_like(x) for x in coeffs_mod[3])
    recon_images.append(np.clip(pywt.waverec2(coeffs_mod, 'sym4'), 0, 255).astype(np.uint8))
    coeffs_mod[2] = tuple(np.zeros_like(x) for x in coeffs_mod[2])
    recon_images.append(np.clip(pywt.waverec2(coeffs_mod, 'sym4'), 0, 255).astype(np.uint8))
    coeffs_mod[1] = tuple(np.zeros_like(x) for x in coeffs_mod[1])
    recon_images.append(np.clip(pywt.waverec2(coeffs_mod, 'sym4'), 0, 255).astype(np.uint8))

    show_images([img_dwt] + recon_images,
                ['4层小波分解系数', '原图', '第1层系数置0', '第1,2层系数置0', '第1-3层系数置0', '4层高频全置0 (平滑)'],
                2, 3, "基于小波的图像平滑")


def progressive_reconstruction(img_path):
    """
    渐进式重构：
        提升网络传输体验（逐步清晰化）；可根据分辨率要求确定解码层数
    """
    if not img_path: return
    f = load_grayscale_image(img_path)
    if f is None: return

    # 双正交小波（有损图像压缩标准核心算法）：
    # 线性相位防止图像边缘扭曲，双正交保证波形对称减少边界处理的伪影
    wavelet = 'bior4.4'
    coeffs = pywt.wavedec2(f, wavelet, level=4)
    img_dwt = build_dwt_image(coeffs)

    approx_images = []

    # 保留低频近似系数，高频置零
    c4 = [coeffs[0]] + [tuple(np.zeros_like(x) for x in detail) for detail in coeffs[1:]]
    # 逆向重构
    approx_images.append(np.clip(pywt.waverec2(c4, wavelet), 0, 255).astype(np.uint8))
    c3 = [coeffs[0], coeffs[1]] + [tuple(np.zeros_like(x) for x in detail) for detail in coeffs[2:]]
    approx_images.append(np.clip(pywt.waverec2(c3, wavelet), 0, 255).astype(np.uint8))
    c2 = [coeffs[0], coeffs[1], coeffs[2]] + [tuple(np.zeros_like(x) for x in detail) for detail in coeffs[3:]]
    approx_images.append(np.clip(pywt.waverec2(c2, wavelet), 0, 255).astype(np.uint8))
    c1 = [coeffs[0], coeffs[1], coeffs[2], coeffs[3], tuple(np.zeros_like(x) for x in coeffs[4])]
    approx_images.append(np.clip(pywt.waverec2(c1, wavelet), 0, 255).astype(np.uint8))
    approx_images.append(np.clip(pywt.waverec2(coeffs, wavelet), 0, 255).astype(np.uint8))

    show_images([img_dwt] + approx_images,
                ['4层小波系数', '第4层近似图像', '重构恢复到第3层', '重构恢复到第2层', '重构恢复到第1层',
                 '最终重构图像'],
                2, 3, "小波渐进式重构")


if __name__ == '__main__':
    fast_wavelet_transform()
    wavelet_coefficients_and_edges(os.path.join('data', 'vase.tif'))
    wavelet_image_smoothing(os.path.join('data', 'points.tif'))
    progressive_reconstruction(os.path.join('data', 'strawberries.tif'))