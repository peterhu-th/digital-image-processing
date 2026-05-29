import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import cut


# Randomly Sampled Texture
def quilt_random(sample, out_size, patch_size):
    """
    随机采样块拼接纹理
    """
    out_img = np.zeros((out_size, out_size, sample.shape[2]), dtype=sample.dtype)
    h, w, _ = sample.shape

    # 从左上角开始，平铺采样块直到填满图像
    for i in range(0, out_size, patch_size):
        for j in range(0, out_size, patch_size):
            # 在原图中随机选一个块的左上角起点
            y = np.random.randint(0, h - patch_size)
            x = np.random.randint(0, w - patch_size)
            patch = sample[y:y + patch_size, x:x + patch_size]

            # 处理边缘越界情况
            y_end = min(i + patch_size, out_size)
            x_end = min(j + patch_size, out_size)
            p_h = y_end - i
            p_w = x_end - j

            out_img[i:y_end, j:x_end] = patch[:p_h, :p_w]

    return out_img


# 辅助函数
def ssd_patch(template, mask, sample):
    """
    利用滤波操作高效计算掩码模板的 SSD (平方差和)
    """
    cost = np.zeros((sample.shape[0], sample.shape[1]), dtype=np.float32)

    # 针对每个通道计算 SSD 并累加
    for c in range(sample.shape[2]):
        T = template[:, :, c].astype(np.float32)
        M = mask.astype(np.float32)
        I = sample[:, :, c].astype(np.float32)

        # SSD 公式: sum((M*T)^2) - 2 * filter(I, M*T) + filter(I^2, M)
        term1 = np.sum((M * T) ** 2)
        term2 = cv2.filter2D(I, ddepth=cv2.CV_32F, kernel=M * T)
        term3 = cv2.filter2D(I ** 2, ddepth=cv2.CV_32F, kernel=M)

        cost += (term1 - 2 * term2 + term3)

    return cost


def choose_sample(cost_img, tol, patch_size):
    """
    从代价图像中选择一个代价较小的块
    """
    h, w = cost_img.shape
    half_p = patch_size // 2

    # 提取有效的中心区域（防止采样的块超出原图边界）
    valid_cost = cost_img[half_p:h - half_p, half_p:w - half_p]

    # 将代价矩阵拍平，排序，找出前 tol 个最小的
    flat_idx = np.argsort(valid_cost.flatten())
    # 在前 tol 个中随机选一个
    choice = np.random.randint(min(tol, len(flat_idx)))
    best_idx = flat_idx[choice]

    # 将一维索引转回二维坐标
    y, x = np.unravel_index(best_idx, valid_cost.shape)

    # 加上 offset 还原到在原 sample 图中的中心坐标
    return y + half_p, x + half_p


# Overlapping Patches
def quilt_simple(sample, out_size, patch_size, overlap, tol):
    """
    基于重叠区域 SSD 的纹理拼接
    """
    out_img = np.zeros((out_size, out_size, sample.shape[2]), dtype=sample.dtype)
    half_p = patch_size // 2

    for i in range(0, out_size, patch_size - overlap):
        for j in range(0, out_size, patch_size - overlap):
            y_end = min(i + patch_size, out_size)
            x_end = min(j + patch_size, out_size)
            p_h = y_end - i
            p_w = x_end - j

            # 初始化掩码和模板
            mask = np.zeros((patch_size, patch_size), dtype=np.float32)
            template = np.zeros((patch_size, patch_size, sample.shape[2]), dtype=np.float32)

            # 提取已有的输出图像作为模板
            template[:p_h, :p_w] = out_img[i:y_end, j:x_end]

            # 设定重叠区域的 Mask (值为1)
            if i > 0:  # 顶部有重叠
                mask[:overlap, :p_w] = 1.0
            if j > 0:  # 左侧有重叠
                mask[:p_h, :overlap] = 1.0

            if i == 0 and j == 0:
                # 左上角的第一个块，纯随机
                y = np.random.randint(0, sample.shape[0] - patch_size)
                x = np.random.randint(0, sample.shape[1] - patch_size)
            else:
                # 计算 SSD 代价并选择样本
                cost = ssd_patch(template, mask, sample)
                cy, cx = choose_sample(cost, tol, patch_size)
                y, x = cy - half_p, cx - half_p

            # 复制像素
            patch = sample[y:y + patch_size, x:x + patch_size]
            out_img[i:y_end, j:x_end] = patch[:p_h, :p_w]

    return out_img


# Seam Finding
def quilt_cut(sample, out_size, patch_size, overlap, tol):
    """
    结合接缝寻找消除拼接伪影
    """
    out_img = np.zeros((out_size, out_size, sample.shape[2]), dtype=sample.dtype)
    half_p = patch_size // 2

    for i in range(0, out_size, patch_size - overlap):
        for j in range(0, out_size, patch_size - overlap):
            y_end = min(i + patch_size, out_size)
            x_end = min(j + patch_size, out_size)
            p_h = y_end - i
            p_w = x_end - j

            mask = np.zeros((patch_size, patch_size), dtype=np.float32)
            template = np.zeros((patch_size, patch_size, sample.shape[2]), dtype=np.float32)
            template[:p_h, :p_w] = out_img[i:y_end, j:x_end]

            if i > 0: mask[:overlap, :p_w] = 1.0
            if j > 0: mask[:p_h, :overlap] = 1.0

            if i == 0 and j == 0:
                y = np.random.randint(0, sample.shape[0] - patch_size)
                x = np.random.randint(0, sample.shape[1] - patch_size)
                patch = sample[y:y + patch_size, x:x + patch_size]
                out_img[i:y_end, j:x_end] = patch[:p_h, :p_w]
                continue

            cost = ssd_patch(template, mask, sample)
            cy, cx = choose_sample(cost, tol, patch_size)
            y, x = cy - half_p, cx - half_p
            patch = sample[y:y + patch_size, x:x + patch_size].copy()

            # 默认全选新 patch (全 1)
            seam_mask = np.ones((patch_size, patch_size), dtype=np.int32)

            # 计算差异矩阵 bndcost: sum((out - patch)^2)
            diff = (template - patch) ** 2
            bndcost = np.sum(diff, axis=2)

            if j > 0:  # 寻找垂直缝隙 (左侧重叠)
                # cut 是找水平缝，所以转置后切，再转置回来
                mask_left = cut(bndcost[:p_h, :overlap].T).T
                seam_mask[:p_h, :overlap] = np.logical_and(seam_mask[:p_h, :overlap], mask_left)

            if i > 0:  # 寻找水平缝隙 (顶部重叠)
                mask_top = cut(bndcost[:overlap, :p_w])
                seam_mask[:overlap, :p_w] = np.logical_and(seam_mask[:overlap, :p_w], mask_top)

            # 将 seam_mask 扩展到 3 通道以便相乘
            seam_mask_3d = np.repeat(seam_mask[:, :, np.newaxis], 3, axis=2)

            # 融合：接缝内用 template (旧图)，接缝外用 patch (新图)
            blended_patch = template * (1 - seam_mask_3d) + patch * seam_mask_3d

            out_img[i:y_end, j:x_end] = blended_patch[:p_h, :p_w]

    return out_img


# Texture Transfer
def texture_transfer(sample, patch_size, overlap, tol, guidance_im, alpha):
    """
    根据目标引导图进行纹理迁移
    """
    out_size_h, out_size_w = guidance_im.shape[:2]
    out_img = np.zeros((out_size_h, out_size_w, sample.shape[2]), dtype=sample.dtype)
    half_p = patch_size // 2

    for i in range(0, out_size_h, patch_size - overlap):
        for j in range(0, out_size_w, patch_size - overlap):
            y_end = min(i + patch_size, out_size_h)
            x_end = min(j + patch_size, out_size_w)
            p_h = y_end - i
            p_w = x_end - j

            mask = np.zeros((patch_size, patch_size), dtype=np.float32)
            template = np.zeros((patch_size, patch_size, sample.shape[2]), dtype=np.float32)
            template[:p_h, :p_w] = out_img[i:y_end, j:x_end]

            if i > 0: mask[:overlap, :p_w] = 1.0
            if j > 0: mask[:p_h, :overlap] = 1.0

            # 计算重叠区域代价 (形状同上)
            overlap_cost = ssd_patch(template, mask, sample)

            # 获取当前对应位置的引导图 Patch (目标表现)
            guide_patch = np.zeros((patch_size, patch_size, guidance_im.shape[2]), dtype=np.float32)
            guide_patch[:p_h, :p_w] = guidance_im[i:y_end, j:x_end]
            guide_mask = np.ones((patch_size, patch_size), dtype=np.float32)

            # 计算与引导图的代价
            target_cost = ssd_patch(guide_patch, guide_mask, sample)

            # 加权组合最终代价
            total_cost = alpha * target_cost + (1 - alpha) * overlap_cost

            cy, cx = choose_sample(total_cost, tol, patch_size)
            y, x = cy - half_p, cx - half_p
            patch = sample[y:y + patch_size, x:x + patch_size].copy()

            # Seam Finding 逻辑
            seam_mask = np.ones((patch_size, patch_size), dtype=np.int32)
            diff = (template - patch) ** 2
            bndcost = np.sum(diff, axis=2)

            if j > 0:
                mask_left = cut(bndcost[:p_h, :overlap].T).T
                seam_mask[:p_h, :overlap] = np.logical_and(seam_mask[:p_h, :overlap], mask_left)
            if i > 0:
                mask_top = cut(bndcost[:overlap, :p_w])
                seam_mask[:overlap, :p_w] = np.logical_and(seam_mask[:overlap, :p_w], mask_top)

            seam_mask_3d = np.repeat(seam_mask[:, :, np.newaxis], 3, axis=2)
            blended_patch = template * (1 - seam_mask_3d) + patch * seam_mask_3d
            out_img[i:y_end, j:x_end] = blended_patch[:p_h, :p_w]

    return out_img


def load_image(filepath):
    """辅助函数：加载图片并归一化"""
    if not os.path.exists(filepath):
        print(f"Error: 找不到 {filepath}。")
        return None
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def run_part1_random(sample_img, out_size, patch_size):
    print("Running Part I: Random Quilt...")
    res = quilt_random(sample_img, out_size, patch_size)
    plt.imshow(res)
    plt.title('Random Quilt')
    plt.show()
    return res


def run_part2_simple(sample_img, out_size, patch_size, overlap, tol):
    print("Running Part II: Simple Quilt (Overlap + SSD)...")
    res = quilt_simple(sample_img, out_size, patch_size, overlap, tol)
    plt.imshow(res)
    plt.title('Simple Quilt (Overlap + SSD)')
    plt.show()
    return res


def run_part3_cut(sample_img, out_size, patch_size, overlap, tol):
    print("Running Part III: Cut Quilt (Seam Finding)...")
    res = quilt_cut(sample_img, out_size, patch_size, overlap, tol)
    plt.imshow(res)
    plt.title('Cut Quilt (Seam Finding)')
    plt.show()
    return res


def run_part4_transfer(sample_img, guidance_img, patch_size, overlap, tol, alpha):
    print("Running Part IV: Texture Transfer...")
    res = texture_transfer(sample_img, patch_size, overlap, tol, guidance_img, alpha)
    plt.imshow(res)
    plt.title('Texture Transfer')
    plt.show()
    return res


if __name__ == '__main__':
    img_dir = 'pictures'
    sample_img_path = os.path.join(img_dir, 'bricks_small.jpg')
    guidance_img_path = os.path.join(img_dir, 'feynman.tiff')
    sample_img = load_image(sample_img_path)

    if sample_img is not None:
        plt.figure(figsize=(4, 4))
        plt.imshow(sample_img)
        plt.title('Original Sample')
        plt.show()

        OUT_SIZE = 200
        PATCH_SIZE = 25
        OVERLAP = 11
        TOL = 3

        run_part1_random(sample_img, OUT_SIZE, PATCH_SIZE)
        run_part2_simple(sample_img, OUT_SIZE, PATCH_SIZE, OVERLAP, TOL)
        run_part3_cut(sample_img, OUT_SIZE, PATCH_SIZE, OVERLAP, TOL)

        guidance_img = load_image(guidance_img_path)
        if guidance_img is not None:
            ALPHA = 0.5
            run_part4_transfer(sample_img, guidance_img, PATCH_SIZE, OVERLAP, TOL, ALPHA)