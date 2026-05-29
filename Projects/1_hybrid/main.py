import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils


# Hybrid Images
def hybridImage(im1, im2, sigma_low, sigma_high):
    '''
    核心算法：生成混合图像
    im1: 低频图像 (离得远看的内容)
    im2: 高频图像 (离得近看的内容)
    '''
    # 低通滤波
    kernel_half_low = int(3 * sigma_low)
    gauss_kernel_low = utils.gaussian_kernel(sigma_low, kernel_half_low)
    low_pass = cv2.filter2D(im1, -1, gauss_kernel_low)

    # 高通滤波
    kernel_half_high = int(3 * sigma_high)
    gauss_kernel_high = utils.gaussian_kernel(sigma_high, kernel_half_high)
    low_pass_im2 = cv2.filter2D(im2, -1, gauss_kernel_high)
    high_pass = im2 - low_pass_im2

    # 混合图像
    hybrid = low_pass + high_pass
    hybrid = np.clip(hybrid, 0.0, 1.0)

    return hybrid, low_pass, high_pass


def show_fft(image, title, ax):
    """辅助函数：利用 utils.py 的 plot_spectrum 显示 2D 傅里叶变换的幅值谱"""
    # 计算频域的幅值谱
    magnitude_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image)))

    # 将当前绘制目标设置为传入的子图 (ax)
    plt.sca(ax)
    # 调用 utils 中的频谱可视化函数
    utils.plot_spectrum(magnitude_spectrum)
    ax.set_title(title)
    ax.axis('off')


def run_part1():
    im1_file = 'pictures/nutmeg.jpg'
    im2_file = 'pictures/DerekPicture.jpg'

    # 读取并归一化图像，提供给交互选点工具
    im1 = np.float32(cv2.imread(im1_file, cv2.IMREAD_GRAYSCALE) / 255.0)
    im2 = np.float32(cv2.imread(im2_file, cv2.IMREAD_GRAYSCALE) / 255.0)

    # 交互式选择眼睛进行对齐
    print("请在弹出的窗口中依次点击两只眼睛进行对齐...")

    # 选择图片
    pts_im1 = utils.prompt_eye_selection(im1)
    plt.show()
    pts_im2 = utils.prompt_eye_selection(im2)
    plt.show()

    # 依据点位对齐图像
    im1_aligned, im2_aligned = utils.align_images(im1_file, im2_file, pts_im1, pts_im2, save_images=False)

    # 把 uint8 格式的 BGR 图像转换为灰度图并归一化到 [0, 1]
    im1_gray = cv2.cvtColor(im1_aligned, cv2.COLOR_BGR2GRAY) / 255.0
    im2_gray = cv2.cvtColor(im2_aligned, cv2.COLOR_BGR2GRAY) / 255.0

    # 设置 sigma 并生成混合图像 (请根据实际图片效果微调这两个参数)
    sigma_low = 15
    sigma_high = 10
    im_hybrid, low_pass, high_pass = hybridImage(im1_gray, im2_gray, sigma_low, sigma_high)

    # 显示最终生成的混合图像
    plt.figure(figsize=(8, 8))
    utils.plot(im_hybrid)  # 直接使用 utils 的 plot 封装函数
    plt.title(f'Hybrid Image (s_low={sigma_low}, s_high={sigma_high})')
    plt.show()

    # 频率分析可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    show_fft(im1_gray, 'Input 1 (FFT)', axes[0, 0])
    show_fft(im2_gray, 'Input 2 (FFT)', axes[1, 0])
    show_fft(low_pass, 'Low-pass Filtered (FFT)', axes[0, 1])
    show_fft(high_pass, 'High-pass Filtered (FFT)', axes[1, 1])
    show_fft(im_hybrid, 'Hybrid Image (FFT)', axes[0, 2])

    # 隐藏右下角空白子图
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


# Image Enhancement
def run_part2_contrast():
    print("--- Running Part II: Contrast Enhancement ---")
    img_poor = cv2.imread('poor_contrast.jpg', cv2.IMREAD_GRAYSCALE)
    if img_poor is None:
        print("未找到 poor_contrast.jpg，跳过对比度增强。")
        return

    # 直方图均衡化增强对比度
    img_eq = cv2.equalizeHist(img_poor)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_poor, cmap='gray'), axes[0].set_title('Original Poor Contrast'), axes[0].axis('off')
    axes[1].imshow(img_eq, cmap='gray'), axes[1].set_title('Contrast Enhanced (Hist Eq)'), axes[1].axis('off')
    plt.show()


def run_part2_color():
    print("--- Running Part II: Color Enhancement ---")
    img_color = cv2.imread('dull_color.jpg')
    if img_color is None:
        print("未找到 dull_color.jpg，跳过色彩增强。")
        return

    img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    # 转换到 HSV 色彩空间进行操作
    hsv_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    # 提升 S 通道 (饱和度)
    s_enhanced = np.clip(s.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)

    hsv_enhanced = cv2.merge([h, s_enhanced, v])
    img_enhanced_rgb = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_color_rgb), axes[0].set_title('Original Color'), axes[0].axis('off')
    axes[1].imshow(img_enhanced_rgb), axes[1].set_title('Enhanced Color (S*1.5)'), axes[1].axis('off')
    plt.show()


def run_part2_color_shift():
    print("--- Running Part II: Color Shift ---")
    img_base = cv2.imread('base_color.jpg')
    if img_base is None:
        print("未找到 base_color.jpg，跳过颜色偏移。")
        return

    img_base_rgb = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)

    # 转换到 LAB 颜色空间
    lab_img = cv2.cvtColor(img_base, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)

    # (a) 更红：增加 A 通道
    a_red = np.clip(a.astype(np.int32) + 30, 0, 255).astype(np.uint8)
    img_more_red = cv2.cvtColor(cv2.merge([l, a_red, b]), cv2.COLOR_LAB2RGB)

    # (b) 更少黄 (偏蓝)：减小 B 通道
    b_less_yellow = np.clip(b.astype(np.int32) - 40, 0, 255).astype(np.uint8)
    img_less_yellow = cv2.cvtColor(cv2.merge([l, a, b_less_yellow]), cv2.COLOR_LAB2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_base_rgb), axes[0].set_title('Original'), axes[0].axis('off')
    axes[1].imshow(img_more_red), axes[1].set_title('(a) More Red (+A)'), axes[1].axis('off')
    axes[2].imshow(img_less_yellow), axes[2].set_title('(b) Less Yellow (-B)'), axes[2].axis('off')
    plt.show()


if __name__ == '__main__':
    run_part1()
    run_part2_contrast()
    run_part2_color()
    run_part2_color_shift()