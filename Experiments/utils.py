import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import heapq

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def show_images(images, titles, rows, cols, figure_title=""):
    """
    统一的图像可视化显示函数（兼容灰度和彩色图像）
    """
    plt.figure(figsize=(4 * cols, 4 * rows))
    plt.suptitle(figure_title, fontsize=16)
    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        img = images[i]

        if len(img.shape) == 2:
            if img.dtype == np.uint8:
                plt.imshow(img, cmap='gray', vmin=0, vmax=255)
            else:
                plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)

        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Exp5
def load_grayscale_image(img_path):
    """读取灰度图"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"找不到图片，请检查路径是否正确: {img_path}")
    return img


# Exp6
def load_color_image(img_path):
    """
    统一的彩色图像读取及校验函数（返回 RGB 格式）
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"找不到图片，请检查路径是否正确: {img_path}")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Exp7
def build_dwt_image(coeffs):
    """
    将多层小波系数拼装成经典的小波系数可视化矩形图
    """
    cA = coeffs[0]
    out_img = cv2.normalize(cA, None, 0, 255, cv2.NORM_MINMAX)

    for detail_level in coeffs[1:]:
        cH, cV, cD = detail_level
        # 高频分量独立归一化
        cH_norm = cv2.normalize(cH, None, 0, 255, cv2.NORM_MINMAX)
        cV_norm = cv2.normalize(cV, None, 0, 255, cv2.NORM_MINMAX)
        cD_norm = cv2.normalize(cD, None, 0, 255, cv2.NORM_MINMAX)
        # 尺寸对齐
        h, w = out_img.shape
        cH_norm = cv2.resize(cH_norm, (w, h))
        cV_norm = cv2.resize(cV_norm, (w, h))
        cD_norm = cv2.resize(cD_norm, (w, h))

        top = np.hstack((out_img, cH_norm))
        bottom = np.hstack((cV_norm, cD_norm))
        out_img = np.vstack((top, bottom))

    return out_img.astype(np.uint8)


# Exp8
def calc_entropy(img):
    """
    计算图像或矩阵的 Shannon 熵：
        H = - Sum (p log_2 p)
    """
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0.0, 256.0))
    p = hist / np.sum(hist)
    p = p[p > 0]
    entropy_val = float(np.sum(p * np.log2(p)))
    return -entropy_val


def huffman_ratio(img):
    """
    构建 Hoffmann 树，求出像素值对应最佳可变长编码，计算实际编码后的理论总比特数，并返回压缩比
    """
    counts = Counter(img.flatten())
    # 单一颜色无需压缩
    if len(counts) <= 1:
        return 1.0

    # 构建 Hoffmann 树 (优先队列)
    heap = [[weight, [symbol, ""]] for symbol, weight in counts.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # 获取每个符号的 Hoffmann 编码长度
    huff_dict = {pair[0]: pair[1] for pair in heap[0][1:]}
    compressed_bits = sum(counts[sym] * len(code) for sym, code in huff_dict.items())

    original_bits = img.size * 8
    if compressed_bits == 0: return float('inf')
    return original_bits / compressed_bits


def mat2lpc(img):
    """
    一维线性预测编码：
        当前像素灰度 - 左边像素灰度，输出误差矩阵，降低数据熵
    """
    img_int = img.astype(np.int32)
    e = np.zeros_like(img_int)
    e[:, 0] = img_int[:, 0]
    e[:, 1:] = img_int[:, 1:] - img_int[:, :-1]
    return e


def mat2gray(img):
    """
    将矩阵线性缩放到 [0, 255]
    """
    img_float = img.astype(np.float64)
    img_min = float(np.min(img_float))
    img_max = float(np.max(img_float))
    if img_max == img_min:
        return np.zeros_like(img, dtype=np.uint8)
    return (255.0 * (img_float - img_min) / (img_max - img_min)).astype(np.uint8)


def igs_quantize(img, bits=4):
    """
    改进的灰度量化，去除心理视觉冗余
    """
    img_flat = img.flatten().astype(np.uint16)
    out_flat = np.zeros_like(img_flat)
    sum_val = 0
    shift = 8 - bits
    lower_mask = (1 << shift) - 1
    upper_mask = 0xFFFF ^ lower_mask
    max_high_val = (1 << bits) - 1
    # 误差累积，进位加给下一个像素
    for i in range(len(img_flat)):
        high = img_flat[i] >> shift
        if high < max_high_val:
            temp = img_flat[i] + sum_val
        else:
            temp = img_flat[i]
        out_flat[i] = temp & upper_mask
        sum_val = temp & lower_mask
    return out_flat.reshape(img.shape).astype(np.uint8)


def calc_rmse(img1, img2):
    """
    计算均方根误差 (RMSE)
    """
    return np.sqrt(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def im2jpeg_simulate(img, quality=95):
    """
    模拟 JPEG 压缩并返回重构图像和压缩比
    """
    success, encoded_img = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
    ratio = img.size / encoded_img.nbytes
    return decoded_img, ratio