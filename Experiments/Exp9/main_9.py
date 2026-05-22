import numpy as np
import cv2
import os
import sys
from skimage.morphology import thin, skeletonize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import show_images, load_grayscale_image
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


def dilation_simple(img_path):
    """
    膨胀的应用
    """
    print("\n--- 膨胀的简单应用 ---")
    if not img_path: return
    A = load_grayscale_image(img_path)
    if A is None: return
    # 确保图像为二值图
    _, A_bin = cv2.threshold(A, 127, 255, cv2.THRESH_BINARY)
    # 定义十字形结构元素
    B = np.array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]], dtype=np.uint8)
    # 执行膨胀运算
    A2 = cv2.dilate(A_bin, B, iterations=1)

    show_images([A_bin, A2], ['原图像', '膨胀后的图像'], 1, 2, "膨胀的简单应用")


def erosion_structural(img_path):
    """
    腐蚀的应用
    """
    print("\n--- 腐蚀的应用 ---")
    if not img_path: return
    A = load_grayscale_image(img_path)
    if A is None: return

    _, A_bin = cv2.threshold(A, 127, 255, cv2.THRESH_BINARY)

    se_10 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))  # 半径为 10
    se_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # 半径为 5
    se_20 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))  # 半径为 20

    A2 = cv2.erode(A_bin, se_10)
    A3 = cv2.erode(A_bin, se_5)  # 保留更多前景
    A4 = cv2.erode(A_bin, se_20)  # 腐蚀最为剧烈

    show_images([A_bin, A2, A3, A4],
                ['原图', '半径10 腐蚀', '半径5 腐蚀', '半径20 腐蚀'],
                2, 2, "基于不同结构元素的腐蚀")


def open_and_close(img_path_logo, img_path_mri):
    """
    开运算和闭运算的应用
    """
    print("\n--- 开运算和闭运算的应用 ---")
    if not img_path_logo or not img_path_mri: return
    f_logo = load_grayscale_image(img_path_logo)
    f_mri = load_grayscale_image(img_path_mri)
    if f_logo is None or f_mri is None: return
    # Logo
    _, f_logo_bin = cv2.threshold(f_logo, 127, 255, cv2.THRESH_BINARY)
    se_20 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))  # 边长为20的方形
    # 开运算：删除细长突出部分和齿状边缘
    fo_logo = cv2.morphologyEx(f_logo_bin, cv2.MORPH_OPEN, se_20)
    # 闭运算：删除指向内部的边缘和小洞
    fc_logo = cv2.morphologyEx(f_logo_bin, cv2.MORPH_CLOSE, se_20)
    # 先开后闭
    foc_logo = cv2.morphologyEx(fo_logo, cv2.MORPH_CLOSE, se_20)

    show_images([f_logo_bin, fo_logo, fc_logo, foc_logo],
                ['原图', '开运算 (去除突出)', '闭运算 (填补小洞)', '先开后闭'],
                2, 2, "开运算与闭运算 - Logo")
    # MRI
    _, f_mri_bin = cv2.threshold(f_mri, 127, 255, cv2.THRESH_BINARY)
    se_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 开运算消除噪声点，引入不连续
    fo_mri = cv2.morphologyEx(f_mri_bin, cv2.MORPH_OPEN, se_3)
    # 闭运算弥补开运算带来的缺口
    foc_mri = cv2.morphologyEx(fo_mri, cv2.MORPH_CLOSE, se_3)

    show_images([f_mri_bin, fo_mri, foc_mri],
                ['原图', '开运算去噪 (出现断裂)', '闭运算修复 (先开后闭)'],
                1, 3, "开闭运算组合 - 指纹修复")


def hit_or_miss_transform(img_path):
    """
    击中或击不中变换
    """
    print("\n--- 击中或击不中变换 ---")
    if not img_path: return
    f = load_grayscale_image(img_path)
    if f is None: return

    _, f_bin = cv2.threshold(f, 127, 255, cv2.THRESH_BINARY)
    kernel = np.array([[-1, -1, -1],
                       [-1, 1, 1],
                       [-1, 1, 0]], dtype=np.int8)
    g = cv2.morphologyEx(f_bin, cv2.MORPH_HITMISS, kernel)
    show_images([f_bin, g], ['原图 (方块阵列)', '击中或击不中变换提取特定角点'], 1, 2, "击中击不中变换")


def thinning_and_skeletonization(img_path_finger, img_path_bone):
    """
    图像细化和骨骼化
    """
    print("\n--- 图像细化和骨骼化 ---")
    if not img_path_finger or not img_path_bone: return
    f_finger = load_grayscale_image(img_path_finger)
    f_bone = load_grayscale_image(img_path_bone)
    if f_finger is None or f_bone is None: return
    # 细化
    _, f_finger_bin = cv2.threshold(f_finger, 127, 255, cv2.THRESH_BINARY)
    bool_finger = f_finger_bin > 0  # skimage 需要布尔型矩阵
    g1 = thin(bool_finger, max_num_iter=1)
    g2 = thin(bool_finger, max_num_iter=2)
    ginf = thin(bool_finger)  # 细化无穷次，达到稳定

    show_images([f_finger_bin, (g1 * 255).astype(np.uint8), (g2 * 255).astype(np.uint8), (ginf * 255).astype(np.uint8)],
                ['指纹原图', '细化 1 次', '细化 2 次', '无穷次细化 (稳定态)'],
                2, 2, "图像细化")
    # 骨骼化
    _, f_bone_bin = cv2.threshold(f_bone, 127, 255, cv2.THRESH_BINARY)
    bool_bone = f_bone_bin > 0
    fs = skeletonize(bool_bone)

    show_images([f_bone_bin, (fs * 255).astype(np.uint8)],
                ['骨骼原图', '骨骼化提取拓扑中轴'],
                1, 2, "图像骨骼化")


def connected_components(img_path):
    """
    连通分量标记
    """
    print("\n--- 连通分量标记 ---")
    if not img_path: return
    f = load_grayscale_image(img_path)
    if f is None: return
    # 人工阈值过滤过亮像素
    h = f.copy()
    h[h > 180] = 0
    # 转换为二值图像
    _, h1 = cv2.threshold(h, 92, 255, cv2.THRESH_BINARY)
    # 连通区域标记 (bwlabel)
    num_labels, labels = cv2.connectedComponents(h1, connectivity=8)
    # 伪彩色映射
    # 将 labels 映射到 HSV 空间的色相通道 (Hue) 以产生缤纷的色彩
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # 转换为 BGR 并将背景设置回纯黑
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[labels == 0] = 0

    show_images([f, h, h1, labeled_img],
                ['原图像', '滤去过亮像素 (h)', '二值化图像 (h1)', f'彩色连通标记结果(n={num_labels - 1})'],
                2, 2, "连通分量标记")


def grayscale_morphology(img_path):
    """
    灰度图像形态学运算
    """
    print("\n--- 灰度图像形态学运算 ---")
    if not img_path: return
    f = load_grayscale_image(img_path)
    if f is None: return

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # 灰度图像开闭运算
    fo = cv2.morphologyEx(f, cv2.MORPH_OPEN, se)
    foc = cv2.morphologyEx(fo, cv2.MORPH_CLOSE, se)
    # 对原图进行大津法 (Otsu) 二值化（此时光照不均会导致失败）
    _, fbin = cv2.threshold(f, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    show_images([f, fo, foc, fbin],
                ['原灰度图 (光照不均)', '灰度开运算 (提取背景估算)', '开闭运算 (平滑背景)', '原图直接Otsu二值化(失败)'],
                2, 2, "灰度图像形态学(1)")

    # 顶帽变换 (Top-Hat) 的两种实现方式：
    # 方式一：原图减去开运算结果 (imsubtract)
    f2 = cv2.subtract(f, fo)
    _, f2bin = cv2.threshold(f2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 方式二：直接调用顶帽算子
    f3 = cv2.morphologyEx(f, cv2.MORPH_TOPHAT, se)
    _, f3bin = cv2.threshold(f3, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    show_images([f2, f2bin, f3, f3bin],
                ['原图减开运算 (校正光照)', '校正后 Otsu 二值化(成功)', '直接顶帽运算 (Top-Hat)', '顶帽后 Otsu 二值化'],
                2, 2, "灰度图像形态学(2) - 顶帽变换解决光照不均")


if __name__ == '__main__':
    dilation_simple(os.path.join('data', 'eight.tif'))
    erosion_structural(os.path.join('data', 'pout.tif'))
    open_and_close(os.path.join('data', 'logo.tif'), os.path.join('data', 'mri.tif'))
    hit_or_miss_transform(os.path.join('data', 'dowels.tif'))
    thinning_and_skeletonization(os.path.join('data', 'mri.tif'), os.path.join('data', 'bones.tif'))
    connected_components(os.path.join('data', 'eight.tif'))
    grayscale_morphology(os.path.join('data', 'rice.tif'))