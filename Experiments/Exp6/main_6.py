import numpy as np
import cv2
import os
import sys
from PIL import Image
from skimage import color
from scipy.spatial.distance import cdist

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import show_images, load_color_image


def color_expression_display(img_path):
    """
    彩色图像的显示
    """
    f = load_color_image(img_path)
    if f is None: return

    # 将文件转化为 Numpy 的 uint8 三维数组 (Height, Width, Channels)
    pil_img = Image.fromarray(f)
    # 索引图像：压缩颜色至 K-means 聚类得到的 32 种颜色
    X1 = pil_img.quantize(colors=32, dither=Image.Dither.NONE)
    # Floyd-Steinberg 误差扩散算法：将像素压缩的同时把误差分配给周围像素，平滑过渡
    X2 = pil_img.quantize(colors=32, dither=Image.Dither.FLOYDSTEINBERG)
    # 得到 Numpy 的 二维数组 (Height, Width)
    g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    pil_g = Image.fromarray(g)
    # 二值图像，每个像素占 1 bit，用疏密模拟灰度
    g1 = pil_g.convert('1', dither=Image.Dither.FLOYDSTEINBERG)

    show_images(
        [f, np.array(X1.convert('RGB')), np.array(X2.convert('RGB')), g, np.array(g1)],
        ['原图', '非抖动转为索引图像', '抖动转为索引图像', '转为灰度图像', '仿抖动处理转为二值图像'],
        2, 3, "彩色图像的表达和显示"
    )


def color_space_conversion(img_path):
    """
    色彩空间：
        RGB（三色叠加）：适合硬件显示，但三个通道相关，改变亮度时三通道同时改变；不符合人类直观感知
        HSV（色调 Hue - 饱和度 Saturation - 明度 Value）：适合图像分割和人类感知
        NTSC / YIQ（亮度 Luminance - 同向 In-phase - 正交 Quadrature）：兼容黑白电视机，易压缩节省带宽
    需要归一化的情境：
        超出显示范围：matplotlib 要求数值在 0 - 1 之间
        存在负数通道：由 RGB 矩阵变换得到的色彩空间，通道会产生负数，归一化才能显示
    """
    f = load_color_image(img_path)
    if f is None: return

    # 转为 NTSC 空间
    ntsc_img = color.rgb2yiq(f)

    # 转为 HSV 空间
    hsv_img = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
    hsv_float = hsv_img.astype(np.float32) / 255.0

    # 归一化 NTSC 分量
    ntsc_show = [
        # Y 通道（亮度）
        np.clip(ntsc_img[:, :, 0], 0, 1),
        # I 通道（色调）
        cv2.normalize(ntsc_img[:, :, 1], None, 0, 1, cv2.NORM_MINMAX),
        # Q 通道（饱和度）
        cv2.normalize(ntsc_img[:, :, 2], None, 0, 1, cv2.NORM_MINMAX)
    ]

    show_images(
        [f, ntsc_show[0], ntsc_show[1], ntsc_show[2],
         f, hsv_float[:, :, 0], hsv_float[:, :, 1], hsv_float[:, :, 2]],
        ['原图', 'ntsc亮度(Y)', 'ntsc色调(I)', 'ntsc饱和度(Q)',
         '原图', 'hsv色调(H)', 'hsv饱和度(S)', 'hsv亮度(V)'],
        2, 4, "彩色空间的转换"
    )


def color_smoothing(img_path):
    """
    彩色图像平滑处理
    """
    f = load_color_image(img_path)
    if f is None: return

    # 转换到 HSV
    hsvimg = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsvimg)

    # 均值滤波：只对明度滤波，人眼对明度敏感；保留色相和饱和度防止颜色污染
    """
    边界填充的 borderType：
    cv2.BORDER_REPLICATE：复制边缘像素
    cv2.BORDER_REFLECT：镜像反射
    cv2.BORDER_CONSTANT：常量填充，常用纯黑导致边缘暗晕
    """
    V_smoothed = cv2.blur(V, (25, 25), borderType=cv2.BORDER_REPLICATE)

    hsv_merged = cv2.merge([H, S, V_smoothed])
    # 转回 RGB
    g1 = cv2.cvtColor(hsv_merged, cv2.COLOR_HSV2RGB)

    # 拉普拉斯锐化
    lapmask = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    g2 = cv2.filter2D(f.astype(np.float32), -1, lapmask, borderType=cv2.BORDER_REPLICATE)
    # 截断保护减法
    sharp = np.clip(f.astype(np.float32) - g2, 0, 255).astype(np.uint8)
    g2_display = cv2.convertScaleAbs(g2)

    show_images(
        [f, g1, g2_display, sharp],
        ['原图', '彩色图像平滑处理 (V分量)', '拉普拉斯算子处理', '锐化处理'],
        2, 2, "彩色图像平滑处理"
    )


def color_edge_detection(img_path):
    """
    彩色图像边缘检测
    """
    f = load_color_image(img_path)
    if f is None: return

    # 获取各通道梯度
    # Sobel 算子：离散的一阶差分，用于计算图像灰度变化率
    f_float = f.astype(np.float32)
    Rx = cv2.Sobel(f_float[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
    Ry = cv2.Sobel(f_float[:, :, 0], cv2.CV_32F, 0, 1, ksize=3)
    Gx = cv2.Sobel(f_float[:, :, 1], cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(f_float[:, :, 1], cv2.CV_32F, 0, 1, ksize=3)
    Bx = cv2.Sobel(f_float[:, :, 2], cv2.CV_32F, 1, 0, ksize=3)
    By = cv2.Sobel(f_float[:, :, 2], cv2.CV_32F, 0, 1, ksize=3)

    # 计算点积：每个元素平方
    gxx = Rx ** 2 + Gx ** 2 + Bx ** 2
    gyy = Ry ** 2 + Gy ** 2 + By ** 2
    gxy = Rx * Ry + Gx * Gy + Bx * By
    # 计算彩色梯度角度
    theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
    # 彩色梯度 VG (Vector Gradient)，考虑通道间关系
    VG = np.sqrt(0.5 * ((gxx + gyy) + (gxx - gyy) * np.cos(2 * theta) + 2 * gxy * np.sin(2 * theta)))
    # 分量梯度简单求和 PPG (Per-Plane Gradient)
    PPG = np.sqrt(Rx ** 2 + Ry ** 2) + np.sqrt(Gx ** 2 + Gy ** 2) + np.sqrt(Bx ** 2 + By ** 2)

    show_images(
        [f, theta, VG, PPG, np.abs(VG - PPG)],
        ['原图', '彩色梯度角度', '彩色梯度 (VG)', '分量梯度 (PPG)', '两种梯度之差'],
        2, 3, "彩色图像边缘检测"
    )


def color_segmentation(img_path):
    """
    彩色图像分割
    """
    f = load_color_image(img_path)
    if f is None: return

    H_img, W_img = f.shape[:2]
    # 创建掩膜矩阵，实现局部操作
    mask = np.zeros((H_img, W_img), dtype=np.uint8)
    # 取中间部分作为 Region of Interest，划定中心和大小
    mask[H_img // 2 - 30:H_img // 2 + 30, W_img // 2 - 30:W_img // 2 + 30] = 1
    g = cv2.bitwise_and(f, f, mask=mask)

    # 基于颜色相似度的区域分割
    # 取样，计算均值
    pixels = f[mask == 1].astype(np.float64)
    if len(pixels) == 0: return
    m = np.mean(pixels, axis=0)
    C_inv = np.linalg.pinv(np.cov(pixels, rowvar=False))
    f_flat = f.astype(np.float64).reshape(-1, 3)

    # 欧氏距离
    D_euc = cdist(f_flat, [m], metric='euclidean').reshape(H_img, W_img)
    E25 = np.zeros((H_img, W_img), dtype=np.uint8)
    E25[D_euc <= 25] = 255
    # 马氏距离：自动适应颜色分布形状
    D_mah = cdist(f_flat, [m], metric='mahalanobis', VI=C_inv).reshape(H_img, W_img)
    M25 = np.zeros((H_img, W_img), dtype=np.uint8)
    M25[D_mah <= 5] = 255

    show_images(
        [f, g, E25, M25],
        ['原图', '选定区域', '欧氏距离彩色分割', '马氏距离彩色分割'],
        2, 2, "彩色图像分割"
    )


def color_space_smoothing_comparison(img_path):
    """
    不同空间平滑对比
    """
    f = load_color_image(img_path)
    if f is None: return

    # 平滑 RGB 三个分量
    rgb_smoothed = cv2.blur(f, (25, 25), borderType=cv2.BORDER_REPLICATE)
    # 在 HSV 空间仅平滑亮度
    hsvimg = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsvimg)
    V_smoothed = cv2.blur(V, (25, 25), borderType=cv2.BORDER_REPLICATE)
    hsv_merged = cv2.merge([H, S, V_smoothed])
    hsv_smoothed_rgb = cv2.cvtColor(hsv_merged, cv2.COLOR_HSV2RGB)

    # 计算两种操作方法的图像差值
    diff = cv2.absdiff(rgb_smoothed, hsv_smoothed_rgb)
    # 放大差异
    diff_amplified = cv2.convertScaleAbs(diff, alpha=5)

    show_images(
        [f, rgb_smoothed, hsv_smoothed_rgb, diff_amplified],
        ['原图', '直接对RGB平滑', '对HSV亮度(V)分量平滑', '两种方法的差异'],
        2, 2, "不同空间平滑对比"
    )


if __name__ == '__main__':
    color_expression_display(os.path.join('data', 'fish.jpg'))
    color_space_conversion(os.path.join('data', 'football.jpg'))
    color_smoothing(os.path.join('data', 'flowers.jpg'))
    color_edge_detection(os.path.join('data', 'fish.jpg'))
    color_segmentation(os.path.join('data', 'football.jpg'))
    color_space_smoothing_comparison(os.path.join('data', 'kids.tif'))