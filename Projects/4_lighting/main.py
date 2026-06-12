import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    read_image, write_image, write_hdr_image,
    gsolve, get_equirectangular_image
)


def display_hdr_image(im_hdr, title="HDR Image"):
    # 使用简单的 Tone Mapping 将 HDR 映射到 0-1 范围以供显示
    img_out = im_hdr / (1.0 + im_hdr)
    img_out = np.clip(img_out, 0, 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_out)
    plt.title(title)
    plt.axis('off')
    plt.show()


def load_ldr_sequence(imdir, imfns):
    """加载多曝光图像序列，并统一缩放至正方形尺寸"""
    ldr_images = []
    imsize = None

    for f in range(len(imfns)):
        im_path = str(os.path.join(imdir, imfns[f]))
        if not os.path.exists(im_path):
            raise FileNotFoundError(f"Error: 找不到样本文件 {im_path}，请准备好测试数据。")

        im = read_image(im_path)
        if f == 0:
            imsize = int((im.shape[0] + im.shape[1]) / 2)
            ldr_images = np.zeros((len(imfns), imsize, imsize, 3), dtype=np.float32)
        ldr_images[f] = cv2.resize(im, (imsize, imsize))

    return ldr_images


# Part I: Naive LDR Merging
def make_hdr_naive(ldr_images, exposures):
    N, H, W, C = ldr_images.shape
    hdr_image = np.zeros((H, W, C), dtype=np.float32)
    log_irradiances = np.zeros((N, H, W, C), dtype=np.float32)

    for i in range(N):
        irrad = ldr_images[i] / exposures[i]
        hdr_image += irrad
        log_irradiances[i] = np.log(irrad + 1e-8)

    hdr_image /= N
    return hdr_image, log_irradiances


# Part II: Weighted LDR Merging
def make_hdr_weighted(ldr_images, exposure_times):
    N, H, W, C = ldr_images.shape
    hdr_image = np.zeros((H, W, C), dtype=np.float32)
    weight_sum = np.zeros((H, W, C), dtype=np.float32) + 1e-8

    for i in range(N):
        # 权重函数 w = 1 - abs( intensity - 0.5 ) * 2
        w = np.clip(1.0 - np.abs(ldr_images[i] - 0.5) * 2.0, 0, 1)
        irrad = ldr_images[i] / exposure_times[i]
        hdr_image += w * irrad
        weight_sum += w

    hdr_image /= weight_sum
    return hdr_image


# Part III: LDR merging with CRF estimation
def make_hdr_estimation(ldr_images, exposure_times, lm):
    N, H, W, C = ldr_images.shape

    # 1) 在镜像球区域内随机采样像素
    y, x = np.ogrid[:H, :W]
    r = min(H, W) // 2
    mask = (x - W // 2) ** 2 + (y - H // 2) ** 2 <= (r * 0.9) ** 2  # 略微缩小半径避免边缘
    valid_pixels = np.where(mask)
    num_valid = len(valid_pixels[0])

    np.random.seed(42)
    sample_indices = np.random.choice(num_valid, min(1000, num_valid), replace=False)
    sample_y = valid_pixels[0][sample_indices]
    sample_x = valid_pixels[1][sample_indices]

    Z = (ldr_images[:, sample_y, sample_x, :] * 255.0).astype(int)
    B = np.log(exposure_times)

    # 权重函数 (映射 intensity 到 weight)
    w_func = np.array([min(z, 255 - z) for z in range(256)])

    g_all = np.zeros((3, 256))
    hdr_image = np.zeros((H, W, C), dtype=np.float32)
    log_irradiances = np.zeros((N, H, W, C), dtype=np.float32)

    # 对 RGB 三个通道分别计算
    for c in range(3):
        Z_c = Z[:, :, c]
        g, _ = gsolve(Z_c, B, lm, w_func)
        g_all[c] = g

        img_c_255 = (ldr_images[..., c] * 255).astype(int)
        sum_w = np.zeros((H, W)) + 1e-8
        sum_w_E = np.zeros((H, W))

        for i in range(N):
            w_img = w_func[img_c_255[i]]
            g_img = g[img_c_255[i]]
            ln_E = g_img - B[i]
            sum_w += w_img
            sum_w_E += w_img * ln_E
            log_irradiances[i, ..., c] = ln_E

        hdr_image[..., c] = np.exp(sum_w_E / sum_w)

    return hdr_image, log_irradiances, g_all


# Part IV: Panoramic Transformations
def panoramic_transform(hdr_image):
    H, W, C = hdr_image.shape

    x = np.linspace(-1, 1, W)
    y = np.linspace(1, -1, H)
    X, Y = np.meshgrid(x, y)

    r2 = X ** 2 + Y ** 2
    mask = r2 <= 1.0

    Z = np.zeros_like(X)
    Z[mask] = np.sqrt(1.0 - r2[mask])

    # 法线向量 N
    N = np.stack((X, Y, Z), axis=-1)
    N[~mask] = 0

    # 视点向量 V
    V = np.array([0.0, 0.0, 1.0])

    # 反射向量 R = V - 2 * dot(V, N) * N
    V_dot_N = N[:, :, 2]  # 因为 V 只有 Z 分量
    R = V - 2 * V_dot_N[..., np.newaxis] * N
    R[~mask] = 0

    return get_equirectangular_image(R, hdr_image)


def run_naive_experiment(ldr_images, exposure_times, out_path):
    print("Computing Naive HDR...")
    naive_hdr_image, _ = make_hdr_naive(ldr_images, exposure_times)
    write_hdr_image(naive_hdr_image, out_path)
    display_hdr_image(naive_hdr_image, "Naive HDR")

def run_weighted_experiment(ldr_images, exposure_times, out_path):
    print("Computing Weighted HDR...")
    weighted_hdr_image = make_hdr_weighted(ldr_images, exposure_times)
    write_hdr_image(weighted_hdr_image, out_path)
    display_hdr_image(weighted_hdr_image, "Weighted HDR")

def run_calibrated_experiment(ldr_images, exposure_times, lm, out_path):
    print("Computing HDR with CRF Estimation...")
    calib_hdr_image, _, _ = make_hdr_estimation(ldr_images, exposure_times, lm)
    write_hdr_image(calib_hdr_image, out_path)
    display_hdr_image(calib_hdr_image, "Calibrated HDR")
    return calib_hdr_image

def run_panoramic_experiment(calib_hdr_image, out_path):
    print("Applying Panoramic Transform...")
    eq_image = panoramic_transform(calib_hdr_image)
    write_hdr_image(eq_image, out_path)
    display_hdr_image(eq_image, "Equirectangular HDR")

def run_compositing_experiment(obj_path, empty_path, mask_path, bg_path, out_path):
    print("Compositing Synthetic Objects...")
    try:
        O = read_image(obj_path)
        E = read_image(empty_path)
        M = read_image(mask_path)
        M = M > 0.5

        if os.path.exists(bg_path):
            I = read_image(bg_path)
            I = cv2.resize(I, (M.shape[1], M.shape[0]))

            # 差分渲染
            diff = O - E
            background_with_shadows = np.clip(I + diff, 0, 1)
            result = np.where(M, O, background_with_shadows)

            plt.figure(figsize=(8, 8))
            plt.imshow(result)
            plt.title("Final Composite")
            plt.axis('off')
            plt.show()
            write_image(result, out_path)
        else:
            print(f"背景图像缺失：{bg_path}")
    except Exception as e:
        print(f"合成步骤异常：未找到对应的图像或读取失败，错误信息: {e}")


if __name__ == '__main__':
    INPUT_SAMPLES_DIR = 'samples'
    INPUT_BLENDER_DIR = 'images'
    OUTPUT_DIR = 'images/outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    IMFNS = ['0024.jpg', '0060.jpg', '0120.jpg', '0205.jpg', '0553.jpg']
    EXPOSURE_TIMES = [1 / 24.0, 1 / 60.0, 1 / 120.0, 1 / 205.0, 1 / 553.0]
    LAMBDA_PARAM = 5
    COMP_OBJ_PATH = os.path.join(INPUT_BLENDER_DIR, 'objects.png')
    COMP_EMPTY_PATH = os.path.join(INPUT_BLENDER_DIR, 'empty.png')
    COMP_MASK_PATH = os.path.join(INPUT_BLENDER_DIR, 'mask.png')
    COMP_BG_PATH = os.path.join(INPUT_SAMPLES_DIR, 'empty.jpg')

    # 加载 LDR 数据序列
    ldr_sequence = load_ldr_sequence(INPUT_SAMPLES_DIR, IMFNS)

    # 朴素法
    run_naive_experiment(ldr_sequence, EXPOSURE_TIMES, os.path.join(OUTPUT_DIR, 'naive_hdr.hdr'))

    # 权重法
    run_weighted_experiment(ldr_sequence, EXPOSURE_TIMES, os.path.join(OUTPUT_DIR, 'weighted_hdr.hdr'))

    # CRF 相机响应函数估计法
    calibrated_hdr = run_calibrated_experiment(ldr_sequence, EXPOSURE_TIMES, LAMBDA_PARAM, os.path.join(OUTPUT_DIR, 'calib_hdr.hdr'))

    # 全景转换
    run_panoramic_experiment(calibrated_hdr, os.path.join(OUTPUT_DIR, 'equirectangular.hdr'))

    # 合成虚拟物体
    run_compositing_experiment(COMP_OBJ_PATH, COMP_EMPTY_PATH, COMP_MASK_PATH, COMP_BG_PATH, os.path.join(OUTPUT_DIR, 'final_composite.png'))