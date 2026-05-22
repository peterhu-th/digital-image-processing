import numpy as np
import cv2
import os
import sys

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import show_images, load_grayscale_image


def point_and_line_detection(img_camera, img_crane):
    """
    二值图像点检测和线检测
    """
    print("\n--- 二值图像点检测和线检测 ---")
    if not img_camera or not img_crane: return

    # 点检测
    f1 = load_grayscale_image(img_camera)
    if f1 is not None:
        # 点测量模板 (拉普拉斯扩展)
        W_point = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]], dtype=np.float32)
        # 空间滤波并取绝对值
        g1 = np.abs(cv2.filter2D(f1.astype(np.float32), -1, W_point))
        # 寻找最大值进行阈值截断 (仅保留孤立的亮点)
        T1 = np.max(g1)
        h1 = (g1 >= T1).astype(np.uint8) * 255

        show_images([f1, cv2.convertScaleAbs(g1), h1],
                    ['原图', '点检测滤波结果', '最大值阈值截断'], 1, 3, "点检测")

        # 单方向线检测 (-45度)
        W_line_45 = np.array([[2, -1, -1],
                              [-1, 2, -1],
                              [-1, -1, 2]], dtype=np.float32)
        g2 = np.abs(cv2.filter2D(f1.astype(np.float32), -1, W_line_45))
        T2 = np.max(g2)
        j2 = (g2 >= T2).astype(np.uint8) * 255

        show_images([f1, cv2.convertScaleAbs(g2), j2],
                    ['原图', '-45度线检测滤波', '线检测边缘提取'], 1, 3, "单方向线检测")

    # 多方向线检测综合
    f2 = load_grayscale_image(img_crane)
    if f2 is not None:
        h1 = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)  # 水平
        h2 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)  # 45度
        h3 = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)  # 垂直
        h4 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32)  # -45度
        J1 = cv2.filter2D(f2.astype(np.float32), -1, h1)
        J2 = cv2.filter2D(f2.astype(np.float32), -1, h2)
        J3 = cv2.filter2D(f2.astype(np.float32), -1, h3)
        J4 = cv2.filter2D(f2.astype(np.float32), -1, h4)
        # 叠加所有方向的线检测特征
        J_sum = np.abs(J1) + np.abs(J2) + np.abs(J3) + np.abs(J4)
        J_sum_norm = cv2.normalize(J_sum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        show_images([f2, J_sum_norm], ['原图', '多方向线特征叠加综合'], 1, 2, "线检测综合")


def sobel_and_canny_detection(img_path):
    """
    [2 & 3] Canny 和 Sobel 算子边缘检测
    """
    print("\n--- [2 & 3] Canny 和 Sobel 算子 ---")
    if not img_path: return
    f = load_grayscale_image(img_path)
    if f is None: return

    # 边缘检测
    canny_default = cv2.Canny(f, 100, 200)
    canny_low_thresh = cv2.Canny(f, 30, 80)
    canny_high_thresh = cv2.Canny(f, 150, 250)

    show_images([f, canny_default, canny_low_thresh, canny_high_thresh],
                ['原图', 'Canny (默认阈值)', 'Canny (低阈值, 细节多)', 'Canny (高阈值, 边缘少)'],
                2, 2, "Canny 算子边缘检测")

    # Sobel 边缘检测
    # 计算水平梯度(垂直边缘)和垂直梯度(水平边缘)
    sobel_x = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    # 获取各个方向的绝对幅值并转回 uint8
    abs_x = cv2.convertScaleAbs(sobel_x)
    abs_y = cv2.convertScaleAbs(sobel_y)
    # 综合幅值 (both)
    sobel_both = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    # Otsu 大津法自动寻找最优阈值二值化 (模拟 MATLAB 的 edge 函数自动行为)
    _, sobel_bin = cv2.threshold(sobel_both, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    show_images([f, sobel_both, abs_x, abs_y, sobel_bin],
                ['原图', 'Sobel 梯度幅值 (Both)', 'Sobel 垂直边缘 (X梯度)', 'Sobel 水平边缘 (Y梯度)',
                 'Sobel 大津法二值化边缘'],
                2, 3, "Sobel 算子边缘检测")


def prewitt_roberts_log_detection(img_path):
    """
    [4, 5, 6] Prewitt, Roberts, LoG 算子边缘检测
    """
    print("\n--- [4, 5, 6] Prewitt, Roberts, LoG 算子 ---")
    if not img_path: return
    f = load_grayscale_image(img_path)
    if f is None: return

    # Prewitt 算子
    kernel_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    px = cv2.filter2D(f.astype(np.float32), -1, kernel_prewitt_x)
    py = cv2.filter2D(f.astype(np.float32), -1, kernel_prewitt_y)
    prewitt_mag = cv2.magnitude(px, py)
    prewitt_mag = cv2.normalize(prewitt_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, prewitt_bin = cv2.threshold(prewitt_mag, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Roberts 算子 (2x2 对角线边缘检测)
    kernel_roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    rx = cv2.filter2D(f.astype(np.float32), -1, kernel_roberts_x)
    ry = cv2.filter2D(f.astype(np.float32), -1, kernel_roberts_y)
    roberts_mag = cv2.magnitude(rx, ry)
    roberts_mag = cv2.normalize(roberts_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, roberts_bin = cv2.threshold(roberts_mag, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # LoG 算子 (高斯-拉普拉斯 / 墨西哥帽)
    f_blur = cv2.GaussianBlur(f, (5, 5), 0)
    log = cv2.Laplacian(f_blur, cv2.CV_64F, ksize=3)
    log_mag = np.abs(log)
    log_mag = cv2.normalize(log_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # 二阶导数的边缘提取往往依赖过零点(Zero-crossing)，这里简化为阈值截断
    _, log_bin = cv2.threshold(log_mag, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    show_images([prewitt_mag, prewitt_bin, roberts_mag, roberts_bin, log_mag, log_bin],
                ['Prewitt 梯度幅值', 'Prewitt 边缘', 'Roberts 梯度幅值', 'Roberts 边缘', 'LoG 绝对值', 'LoG 边缘'],
                2, 3, "Prewitt, Roberts, LoG 算子比对")


def global_thresholding(img_path):
    """
    采用全局阈值(Otsu大津法)对图像进行分割
    """
    print("\n--- 全局阈值图像分割 ---")
    if not img_path: return
    I = load_grayscale_image(img_path)
    if I is None: return

    # graythresh 对应 OpenCV 的 THRESH_OTSU，自动寻找最大类间方差阈值
    T, J = cv2.threshold(I, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(f"Otsu 大津法自动计算出的全局阈值为: {T}")

    show_images([I, J], ['原图 (coins)', f'全局阈值二值化分割 (T={T})'], 1, 2, "全局阈值分割")


if __name__ == '__main__':
    point_and_line_detection(os.path.join('data', 'cameraman.tif'), os.path.join('data', 'gantrycrane.png'))
    sobel_and_canny_detection(os.path.join('data', 'cameraman.tif'))
    prewitt_roberts_log_detection(os.path.join('data', 'pigeon.jpg'))
    global_thresholding(os.path.join('data', 'coins.png'))