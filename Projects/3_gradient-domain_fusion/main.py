import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from typing import Optional
import utils


def toy_reconstruct(img: np.ndarray) -> np.ndarray:
    """
    Toy Problem: 通过源图像的梯度值及一个像素的绝对强度重建原图。
    通过构建稀疏矩阵并求解最小二乘问题实现。
    """
    h, w = img.shape
    im2var = np.arange(h * w).reshape((h, w))

    # 构建系数矩阵 A 和目标向量 b
    # 方程数量: (h) * (w-1) (x方向梯度) + (h-1) * (w) (y方向梯度) + 1 (基准点)
    num_eqs = h * (w - 1) + (h - 1) * w + 1
    num_vars = h * w

    A = scipy.sparse.lil_matrix((num_eqs, num_vars))
    b = np.zeros(num_eqs)

    e = 0
    # 最小化 X/Y 方向梯度误差
    for y in range(h):
        for x in range(w - 1):
            A[e, im2var[y, x + 1]] = 1
            A[e, im2var[y, x]] = -1
            b[e] = img[y, x + 1] - img[y, x]
            e += 1

    for y in range(h - 1):
        for x in range(w):
            A[e, im2var[y + 1, x]] = 1
            A[e, im2var[y, x]] = -1
            b[e] = img[y + 1, x] - img[y, x]
            e += 1

    # 添加左上角像素点作为绝对基准点
    A[e, im2var[0, 0]] = 1
    b[e] = img[0, 0]

    # 求解 Ax = b
    A = A.tocsr()
    v = scipy.sparse.linalg.lsqr(A, b)[0]

    return v.reshape((h, w))


def _solve_blend(object_img: np.ndarray, object_mask: np.ndarray, bg_img: np.ndarray, bg_ul: tuple,
                 mixed: bool = False) -> np.ndarray:
    """
    泊松融合与混合融合的底层求解器（单通道）。
    """
    h, w = object_img.shape

    # 获取背景区域的对应 Patch
    bg_patch = bg_img[bg_ul[0]:bg_ul[0] + h, bg_ul[1]:bg_ul[1] + w].copy()

    # 获取需要计算的 Mask 内像素点索引
    y_idx, x_idx = np.nonzero(object_mask)
    num_vars = len(y_idx)

    # 构建坐标到变量索引的映射矩阵 (-1 表示在 Mask 外部)
    im2var = -np.ones((h, w), dtype=int)
    for i in range(num_vars):
        im2var[y_idx[i], x_idx[i]] = i

    A = scipy.sparse.lil_matrix((num_vars, num_vars))
    b = np.zeros(num_vars)

    # 构建拉普拉斯矩阵与目标梯度向量
    for i in range(num_vars):
        y, x = y_idx[i], x_idx[i]
        A[i, i] = 4

        # 遍历四个方向的元素
        neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
        for ny, nx in neighbors:
            if 0 <= ny < h and 0 <= nx < w:
                # 计算梯度
                grad_obj = object_img[y, x] - object_img[ny, nx]
                if mixed:
                    grad_bg = bg_patch[y, x] - bg_patch[ny, nx]
                    # 混合融合：取前景和背景中梯度绝对值较大的一个
                    target_grad = grad_obj if abs(grad_obj) > abs(grad_bg) else grad_bg
                else:
                    target_grad = grad_obj

                if object_mask[ny, nx]:
                    # 邻居在 Mask 内部：将其加入 A 矩阵作为未知数
                    A[i, im2var[ny, nx]] = -1
                    b[i] += target_grad
                else:
                    # 邻居在 Mask 外部：邻居像素值固定为背景值，移至等式右侧
                    b[i] += target_grad + bg_patch[ny, nx]
            else:
                # 边界条件防护
                b[i] += object_img[y, x]

    # 求解稀疏线性方程组
    A = A.tocsr()
    v = scipy.sparse.linalg.spsolve(A, b)

    # 将求解结果填回背景 Patch 中
    blend_patch = bg_patch.copy()
    for i in range(num_vars):
        blend_patch[y_idx[i], x_idx[i]] = np.clip(v[i], 0, 1)  # 限制像素值在 [0, 1] 之间

    return blend_patch


def poisson_blend(object_img: np.ndarray, object_mask: np.ndarray, bg_img: np.ndarray, bg_ul: tuple) -> np.ndarray:
    """泊松融合：保持源图像梯度，无缝融入背景"""
    return _solve_blend(object_img, object_mask, bg_img, bg_ul, mixed=False)


def mixed_blend(object_img: np.ndarray, object_mask: np.ndarray, bg_img: np.ndarray, bg_ul: tuple) -> np.ndarray:
    """混合梯度融合：结合源图像和背景图像的最强梯度（常用于透明或带有纹理的融合）"""
    return _solve_blend(object_img, object_mask, bg_img, bg_ul, mixed=True)


def color_transfer(source_patch: np.ndarray, bg_patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    色彩转移前处理：将源图像在 Mask 区域内的颜色分布（均值和方差）对齐到背景区域
    """
    out_patch = np.zeros_like(source_patch)
    for c in range(3):
        s_ch = source_patch[:, :, c]
        b_ch = bg_patch[:, :, c]

        # 只在 Mask 区域内计算统计量，避免受到无效黑边的干扰
        s_pixels = s_ch[mask > 0]
        b_pixels = b_ch[mask > 0]

        s_mean = float(np.mean(s_pixels))
        s_std = float(np.std(s_pixels))
        b_mean = float(np.mean(b_pixels))
        b_std = float(np.std(b_pixels))

        if s_std < 1e-6:
            s_std = 1e-6

        # 线性色彩映射公式: out = (in - mean_in) * (std_out / std_in) + mean_out
        out_ch = (s_ch - s_mean) * (b_std / s_std) + b_mean
        out_patch[:, :, c] = np.clip(out_ch, 0, 1)

    return out_patch


def load_image(filepath: str, to_gray: bool = False) -> Optional[np.ndarray]:
    """辅助函数：加载图片、转换通道并归一化"""
    if not os.path.exists(filepath):
        print(f"Error: 找不到文件 {filepath}")
        return None
    if to_gray:
        img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    return img.astype('double') / 255.0


def run_toy_problem(toy_img: np.ndarray):
    """运行 Toy Problem 图像重建测试"""
    print("Running Toy Problem...")
    im_out = toy_reconstruct(toy_img)

    print("Max error is: ", np.sqrt(((im_out - toy_img) ** 2).max()))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.title("Original Toy Image"), plt.imshow(toy_img, cmap="gray")
    plt.subplot(1, 2, 2), plt.title("Reconstructed"), plt.imshow(im_out, cmap="gray")
    plt.show()


def run_poisson_blend(background_img: np.ndarray, object_img: np.ndarray, xs: tuple, ys: tuple, bottom_center: tuple):
    """运行泊松融合"""
    print("Running Poisson Blending...")

    # 调用 utils 生成并裁剪粗略遮罩，定位计算区域
    object_mask = utils.get_mask(ys, xs, object_img)
    obj_img_crop, obj_mask_crop = utils.crop_object_img(object_img, object_mask)
    bg_ul = utils.upper_left_background_rc(obj_mask_crop, bottom_center)

    h, w = obj_img_crop.shape[:2]
    im_blend = background_img.copy()

    # 三通道分别调用泊松融合求解器
    for b in range(3):
        patch = poisson_blend(obj_img_crop[:, :, b], obj_mask_crop, background_img[:, :, b].copy(), bg_ul)
        im_blend[bg_ul[0]:bg_ul[0] + h, bg_ul[1]:bg_ul[1] + w, b] = patch

    plt.figure(figsize=(10, 10))
    plt.title("Poisson Blending")
    plt.imshow(im_blend)
    plt.axis('off')
    plt.show()


def run_mixed_blend(background_img: np.ndarray, object_img: np.ndarray, xs: tuple, ys: tuple, bottom_center: tuple):
    """运行混合梯度融合 (使用 utils 工具处理粗略 Mask)"""
    print("Running Mixed Gradients Blending...")

    # 调用 utils 生成并裁剪粗略遮罩，定位计算区域
    object_mask = utils.get_mask(ys, xs, object_img)
    obj_img_crop, obj_mask_crop = utils.crop_object_img(object_img, object_mask)
    bg_ul = utils.upper_left_background_rc(obj_mask_crop, bottom_center)

    h, w = obj_img_crop.shape[:2]
    im_mix = background_img.copy()

    # 三通道分别调用混合梯度融合求解器
    for b in range(3):
        patch = mixed_blend(obj_img_crop[:, :, b], obj_mask_crop, background_img[:, :, b].copy(), bg_ul)
        im_mix[bg_ul[0]:bg_ul[0] + h, bg_ul[1]:bg_ul[1] + w, b] = patch

    plt.figure(figsize=(10, 10))
    plt.title("Mixed Gradients Blending")
    plt.imshow(im_mix)
    plt.axis('off')
    plt.show()


def run_auto_blend(background_img: np.ndarray, object_img: np.ndarray, bottom_center: tuple):
    """运行自动轮廓提取 (GrabCut) 并进行混合梯度融合"""
    print("Running Auto Contour Extraction (GrabCut) + Mixed Blending...")

    img_uint8 = (object_img * 255).astype(np.uint8)

    # 初始化掩码和 GrabCut 所需的背景/前景模型
    mask = np.zeros(img_uint8.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 定义一个包含企鹅的初始矩形框 (x, y, width, height)
    rect = (65, 24, 359 - 65, 457 - 24)

    # 执行 GrabCut 算法 (迭代 5 次)
    cv2.grabCut(img_uint8, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # 二值 Mask：GrabCut 中 1 代表确定前景，3 代表可能前景
    object_mask = np.where((mask == 1) | (mask == 3), 1, 0).astype(np.uint8)

    # 对生成的 Mask 进行一次轻微膨胀，确保边缘平滑过渡
    object_mask = cv2.dilate(object_mask, np.ones((3, 3), np.uint8), iterations=1)

    # 调用 utils 生成并裁剪遮罩，定位计算区域
    obj_img_crop, obj_mask_crop = utils.crop_object_img(object_img, object_mask)
    bg_ul = utils.upper_left_background_rc(obj_mask_crop, bottom_center)

    h, w = obj_img_crop.shape[:2]
    im_mix = background_img.copy()

    # 混合梯度融合
    for b in range(3):
        patch = mixed_blend(obj_img_crop[:, :, b], obj_mask_crop, background_img[:, :, b].copy(), bg_ul)
        im_mix[bg_ul[0]:bg_ul[0] + h, bg_ul[1]:bg_ul[1] + w, b] = patch

    # 可视化对比结果
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.title("Auto Extracted Mask (GrabCut)")
    plt.imshow(object_mask, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Auto Blended Result")
    plt.imshow(im_mix)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    toy_path = 'pictures/toy_problem.png'
    bg_path = 'pictures/im2.JPG'
    obj_path = 'pictures/penguin-chick.jpeg'

    # 集中定义 Mask 顶点坐标与背景目标放置中心点
    xs_coord = (65, 359, 359, 65)
    ys_coord = (24, 24, 457, 457)
    target_bottom_center = (500, 2500)

    # Toy Problem
    toy_image = load_image(toy_path, to_gray=True)
    if toy_image is not None:
        run_toy_problem(toy_image)

    # 图像融合
    bg_image = load_image(bg_path, to_gray=False)
    obj_image = load_image(obj_path, to_gray=False)

    if bg_image is not None and obj_image is not None:
        # 泊松梯度域融合
        run_poisson_blend(bg_image, obj_image, xs_coord, ys_coord, target_bottom_center)
        # 混合梯度域融合
        run_mixed_blend(bg_image, obj_image, xs_coord, ys_coord, target_bottom_center)
        # 自动轮廓提取 + 混合梯度域融合
        run_auto_blend(bg_image, obj_image, target_bottom_center)