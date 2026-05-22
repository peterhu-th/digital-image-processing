import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import utils


def toy_reconstruct(img: np.ndarray) -> np.ndarray:
    """
    Toy Problem: 通过源图像的梯度值及一个像素的绝对强度重建原图。
    通过构建稀疏矩阵并求解最小二乘问题实现。
    """
    h, w = img.shape
    im2var = np.arange(h * w).reshape((h, w))

    # 构建系数矩阵 A 和目标向量 b
    # 方程数量: (h)*(w-1) (x方向梯度) + (h-1)*(w) (y方向梯度) + 1 (基准点)
    num_eqs = h * (w - 1) + (h - 1) * w + 1
    num_vars = h * w

    A = scipy.sparse.lil_matrix((num_eqs, num_vars))
    b = np.zeros(num_eqs)

    e = 0
    # 1. 最小化 X 方向梯度误差
    for y in range(h):
        for x in range(w - 1):
            A[e, im2var[y, x + 1]] = 1
            A[e, im2var[y, x]] = -1
            b[e] = img[y, x + 1] - img[y, x]
            e += 1

    # 2. 最小化 Y 方向梯度误差
    for y in range(h - 1):
        for x in range(w):
            A[e, im2var[y + 1, x]] = 1
            A[e, im2var[y, x]] = -1
            b[e] = img[y + 1, x] - img[y, x]
            e += 1

    # 3. 添加一个绝对基准点（如左上角像素点）
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

        # 遍历四个方向的邻居：上, 下, 左, 右
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


def main():
    # ================= 1. Part 1 Toy Problem =================
    print("Running Toy Problem...")
    toy_path = 'picture/toy_problem.png'
    if os.path.exists(toy_path):
        toy_img = cv2.cvtColor(cv2.imread(toy_path), cv2.COLOR_BGR2GRAY).astype('double') / 255.0

        im_out = toy_reconstruct(toy_img)

        print("Max error is: ", np.sqrt(((im_out - toy_img) ** 2).max()))

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1), plt.title("Original Toy Image"), plt.imshow(toy_img, cmap="gray")
        plt.subplot(1, 2, 2), plt.title("Reconstructed"), plt.imshow(im_out, cmap="gray")
        plt.show()
    else:
        print(f"Warning: {toy_path} not found.")

    # ================= 2. Preparation for Blending =================
    print("Preparing images for Blending...")
    bg_path = 'picture/im2.JPG'
    obj_path = 'picture/penguin-chick.jpeg'

    if not (os.path.exists(bg_path) and os.path.exists(obj_path)):
        print("Background or Object image not found in 'picture/' directory. Exiting blending steps.")
        return

    background_img = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB).astype('double') / 255.0
    object_img = cv2.cvtColor(cv2.imread(obj_path), cv2.COLOR_BGR2RGB).astype('double') / 255.0

    # 预设的 Mask 坐标 (原 Notebook 中的静态坐标)
    xs = (65, 359, 359, 65)
    ys = (24, 24, 457, 457)
    object_mask = utils.get_mask(ys, xs, object_img)
    bottom_center = (500, 2500)  # (x, y) 对应列、行

    object_img, object_mask = utils.crop_object_img(object_img, object_mask)
    bg_ul = utils.upper_left_background_rc(object_mask, bottom_center)

    # ================= 3. Part 2 Poisson Blending =================
    print("Running Poisson Blending...")
    im_blend = np.zeros(background_img.shape)
    for b in range(3):
        im_blend[:, :, b] = poisson_blend(object_img[:, :, b], object_mask, background_img[:, :, b].copy(), bg_ul)

    plt.figure(figsize=(10, 10))
    plt.title("Poisson Blending")
    plt.imshow(im_blend)
    plt.axis('off')
    plt.show()

    # ================= 4. Part 3 Mixed Gradients =================
    print("Running Mixed Gradients Blending...")
    im_mix = np.zeros(background_img.shape)
    for b in range(3):
        im_mix[:, :, b] = mixed_blend(object_img[:, :, b], object_mask, background_img[:, :, b].copy(), bg_ul)

    plt.figure(figsize=(10, 10))
    plt.title("Mixed Gradients Blending")
    plt.imshow(im_mix)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()