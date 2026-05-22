import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    show_images, load_grayscale_image,
    calc_entropy, huffman_ratio, mat2lpc, mat2gray,
    igs_quantize, calc_rmse, im2jpeg_simulate
)


def entropy_and_huffman_coding(img_path):
    """
    计算图像的熵和 Hoffmann 编码：
        熵越小，Hoffmann 压缩比越高
    """
    print("\n--- 计算图像的熵和 Hoffmann 编码 ---")
    # 测试矩阵 Shannon 熵和 Hoffmann 压缩比计算
    f = np.array([[119, 123, 168, 119],
                  [123, 119, 168, 168],
                  [119, 119, 107, 119],
                  [107, 107, 119, 119]], dtype=np.uint8)
    ent_f = calc_entropy(f)
    print(f"f1 的熵: {ent_f:.4f}")
    f2 = np.array([[2, 3, 4, 2],
                   [3, 2, 4, 4],
                   [2, 2, 1, 2],
                   [1, 1, 2, 2]], dtype=np.uint8)
    cr_f2 = huffman_ratio(f2)
    print(f"f2 的 Hoffmann 压缩比: {cr_f2:.4f}")
    # 真实图像测试
    if not img_path: return
    img = load_grayscale_image(img_path)
    if img is None: return
    ent_img = calc_entropy(img)
    cr_img = huffman_ratio(img)
    print(f"[{img_path}] 真实图像熵: {ent_img:.4f} Hoffmann 压缩比 (无损): {cr_img:.4f}")


def interpixel_redundancy_and_predictive_coding(img_path1, img_path2):
    """
    像素间冗余和预测编码（无损压缩）
    """
    print("\n--- 像素间冗余和预测编码 ---")
    if not img_path1 or not img_path2: return
    f1 = load_grayscale_image(img_path1)
    f2 = load_grayscale_image(img_path2)
    if f1 is None or f2 is None: return
    # 验证熵编码局限性
    print(f"图像 1 [{img_path1}] 熵: {calc_entropy(f1):.4f}, 压缩比: {huffman_ratio(f1):.4f}")
    print(f"图像 2 [{img_path2}] 熵: {calc_entropy(f2):.4f}, 压缩比: {huffman_ratio(f2):.4f}")
    # 预测编码
    e = mat2lpc(f2)
    e_gray = mat2gray(e)
    ent_e = calc_entropy(e)
    cr_e = huffman_ratio(e)
    print(f"\n图像 2 线性预测编码后误差矩阵的熵: {ent_e:.4f}, Hoffmann 压缩比: {cr_e:.4f}")

    show_images([f2, e_gray], ['原图', '预测误差图像'], 1, 2, "像素间冗余和预测编码")
    # 绘制直方图
    plt.figure()
    plt.hist(e.flatten(), bins=100, color='black')
    plt.title('预测误差图像直方图')
    plt.show()


def psychovisual_redundancy_and_quantization(img_path):
    """
    心理视觉冗余和量化（有损压缩）
    """
    print("\n--- 心理视觉冗余和量化 ---")
    if not img_path: return
    f = load_grayscale_image(img_path)
    if f is None: return

    # IGS 量化 (位深降至4)
    q = igs_quantize(f, bits=4)
    qs = (q / 16).astype(np.uint8)
    # 对量化数据进行预测编码和压缩
    e = mat2lpc(qs)
    cr_q = huffman_ratio(e)
    print(f"LPC + Hoffmann 压缩比: {cr_q:.4f}")
    # 反量化与误差计算
    nq = (qs * 16).astype(np.uint8)
    rmse = calc_rmse(f, nq)
    print(f"量化导致的均方根误差 (RMSE): {rmse:.4f}")

    show_images([f, q, nq], ['原图', 'IGS量化结果', '反量化重构图像'], 1, 3, "心理视觉冗余和量化")


def jpeg_compression(img_path):
    """
    JPEG 压缩画质对比
    """
    if not img_path: return
    f = load_grayscale_image(img_path)
    if f is None: return
    # 高质量参数模拟 (Quality=90)
    f1, cr1 = im2jpeg_simulate(f, quality=90)
    rmse1 = calc_rmse(f, f1)
    print(f"高质量 JPEG 压缩比: {cr1:.4f}, RMSE: {rmse1:.4f}")
    # 低质量参数模拟 (Quality=10)
    f4, cr4 = im2jpeg_simulate(f, quality=10)
    rmse4 = calc_rmse(f, f4)
    print(f"低质量 JPEG 压缩比: {cr4:.4f}, RMSE: {rmse4:.4f}")

    show_images([f, f1, f4],
                ['原图', f'高质量JPEG(CR:{cr1:.2f})', f'低质量JPEG(CR:{cr4:.2f})'],
                1, 3, "JPEG 有损压缩对比")


if __name__ == '__main__':
    entropy_and_huffman_coding(os.path.join('data', 'shadow.tif'))
    interpixel_redundancy_and_predictive_coding(os.path.join('data', 'cameraman.tif'),os.path.join('data', 'mri.tif'))
    psychovisual_redundancy_and_quantization(os.path.join('data', 'mri.tif'))
    jpeg_compression(os.path.join('data', 'trees.tif'))