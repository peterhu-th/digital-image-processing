import os
import cv2
import numpy as np
import glob
from utils import blendImages, vidwrite_from_numpy


def score_projection(pt1, pt2, thresh=5.0):
    """ 计算 RANSAC 内点得分 (使用欧氏距离) """
    diff = pt1 - pt2
    dist = np.linalg.norm(diff, axis=0)
    inliers = dist < thresh
    score = np.sum(inliers)
    return score, inliers


def computeHomography(pts1, pts2, normalization_func=None):
    """ 使用 SVD 计算单应性矩阵 H (pts2 = H * pts1) """
    num_pts = pts1.shape[1]
    A = np.zeros((2 * num_pts, 9))
    for i in range(num_pts):
        x, y = pts1[0, i], pts1[1, i]
        u, v = pts2[0, i], pts2[1, i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, x * u, y * u, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * v, y * v, v]

    U, S, Vh = np.linalg.svd(A)
    H = Vh[-1, :].reshape(3, 3)
    return H


def auto_homography(Ia, Ib):
    """ 提取特征并使用 RANSAC 计算鲁棒的单应性矩阵 H """
    if Ia.dtype == 'float32': Ia = (Ia * 255).astype(np.uint8)
    if Ib.dtype == 'float32': Ib = (Ib * 255).astype(np.uint8)

    Ia_gray = cv2.cvtColor(Ia, cv2.COLOR_BGR2GRAY)
    Ib_gray = cv2.cvtColor(Ib, cv2.COLOR_BGR2GRAY)

    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        sift = cv2.xfeatures2d.SIFT_create()

    kp_a, des_a = sift.detectAndCompute(Ia_gray, None)
    kp_b, des_b = sift.detectAndCompute(Ib_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a, des_b, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    numMatches = int(len(good))
    if numMatches < 4:
        raise ValueError("Not enough matches found.")

    Xa = np.ones((3, numMatches))
    Xb = np.ones((3, numMatches))

    for idx, match_i in enumerate(good):
        Xa[:, idx][0:2] = kp_a[match_i.queryIdx].pt
        Xb[:, idx][0:2] = kp_b[match_i.trainIdx].pt

    niter = 1000
    best_score = 0
    n_to_sample = 4  # 至少需要 4 对点求解单应性矩阵
    best_H = np.eye(3)

    for t in range(niter):
        subset = np.random.choice(numMatches, n_to_sample, replace=False)
        pts1 = Xa[:, subset]
        pts2 = Xb[:, subset]

        H_t = computeHomography(pts1, pts2)
        Xb_ = np.dot(H_t, Xa)

        score_t, inliers_t = score_projection(Xb[:2, :] / (Xb[2, :] + 1e-8), Xb_[:2, :] / (Xb_[2, :] + 1e-8))

        if score_t > best_score:
            best_score = score_t
            best_H = H_t

    print(f'RANSAC Best Score: {best_score} / {numMatches}')
    return best_H


def stitch_two(im1_path, im2_path, Tr, out_size, output_dir):
    """ Part 1: Stitch two key frames (Frame 270 and 450) """
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)

    H_1to2 = auto_homography(im1, im2)

    # 结合平移矩阵投影到公共画布
    H_proj_1 = np.dot(Tr, H_1to2)
    H_proj_2 = Tr

    warp1 = cv2.warpPerspective(im1, H_proj_1, out_size)
    warp2 = cv2.warpPerspective(im2, H_proj_2, out_size)

    blended = blendImages(warp1, warp2)
    out_path = os.path.join(output_dir, "stitch.jpg")
    cv2.imwrite(out_path, blended)
    print("Saved stitch.jpg")
    return H_1to2


def panorama_five(img_paths, key_indices, Tr, out_size, output_dir):
    """ Part 2: Panorama using five key frames """
    frames = [cv2.imread(img_paths[i - 1]) for i in key_indices]
    ref_idx = 2  # 对应 frame 450 (索引为 2)

    # 计算相邻帧之间的单应性
    H_matrices = {}  # 记录 i 映射到 i+1 的 H
    for i in range(len(frames) - 1):
        print(f"Mapping keyframe {key_indices[i]} to {key_indices[i + 1]}...")
        H_matrices[i] = auto_homography(frames[i], frames[i + 1])

    # 计算所有帧到 450 的单应性
    H_to_ref = {ref_idx: np.eye(3)}

    # 逆推向前的帧 (90->270, 270->450)
    H_to_ref[1] = H_matrices[1]
    H_to_ref[0] = np.dot(H_matrices[1], H_matrices[0])

    # 顺推向后的帧 (630->450, 810->630->450)
    # i+1 映射到 i 需要求逆
    H_3_to_2 = np.linalg.inv(H_matrices[2])
    H_4_to_3 = np.linalg.inv(H_matrices[3])
    H_to_ref[3] = H_3_to_2
    H_to_ref[4] = np.dot(H_3_to_2, H_4_to_3)

    # 投影与混合
    canvas = np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)
    for i in range(len(frames)):
        H_final = np.dot(Tr, H_to_ref[i])
        warp = cv2.warpPerspective(frames[i], H_final, out_size)
        canvas = blendImages(warp, canvas)

    out_path = os.path.join(output_dir, "panorama.jpg")
    cv2.imwrite(out_path, canvas)
    print("Saved panorama.jpg")
    return H_to_ref, frames


def map_video(img_paths, ref_frame_idx, Tr, out_size):
    """ Part 3: Map all frames to the reference plane (Simplified for memory) """
    # 为防止内存溢出，每隔 10 帧抽样一次，也可去掉切片 [::10]
    sample_paths = img_paths[::10]
    ref_img = cv2.imread(img_paths[ref_frame_idx - 1])

    mapped_video = []
    for idx, path in enumerate(sample_paths):
        img = cv2.imread(path)
        # 简单策略：直接求每一帧到 450 帧的单应性 (如果重叠太少理论上应该找最近关键帧)
        try:
            H = auto_homography(img, ref_img)
            H_final = np.dot(Tr, H)
            warp = cv2.warpPerspective(img, H_final, out_size)
            mapped_video.append(warp)
            if idx % 10 == 0: print(f"Mapped {idx + 1}/{len(sample_paths)} frames...")
        except Exception:
            continue  # 特征点不够跳过

    mapped_video_np = np.array(mapped_video)
    # 可以调用 vidwrite_from_numpy(..., mapped_video_np) 保存视频，这里以返回数组为例
    return mapped_video_np


def background_panorama(mapped_video, output_dir):
    """ Part 4: Create background panorama by calculating median """
    # 为了不把黑色背景（值为0）算进中位数，把 0 转成 NaN
    mapped_float = mapped_video.astype(np.float32)
    mapped_float[mapped_float == 0] = np.nan

    # 沿时间轴计算中位数
    print("Computing temporal median, this might take a minute...")
    bg_pano = np.nanmedian(mapped_float, axis=0)

    # 还原成 uint8
    bg_pano = np.nan_to_num(bg_pano).astype(np.uint8)

    out_path = os.path.join(output_dir, "bg_panorama.jpg")
    cv2.imwrite(out_path, bg_pano)
    print("Saved bg_panorama.jpg")
    return bg_pano


def background_movie(bg_pano, img_paths, ref_frame_idx, Tr):
    """ Part 5: Create background movie by inverse projecting panorama to frames """
    ref_img = cv2.imread(img_paths[ref_frame_idx - 1])
    h, w = ref_img.shape[:2]

    sample_paths = img_paths[::10]
    bg_movie = []

    for path in sample_paths:
        img = cv2.imread(path)
        try:
            H_img_to_ref = auto_homography(img, ref_img)
            # 全景图坐标是从 Img 坐标经过 H_final = Tr * H_img_to_ref 变来的
            # 反推: Img = inv(H_final) * Pano
            H_final = np.dot(Tr, H_img_to_ref)
            H_pano_to_img = np.linalg.inv(H_final)

            bg_frame = cv2.warpPerspective(bg_pano, H_pano_to_img, (w, h))
            bg_movie.append(bg_frame)
        except Exception:
            continue

    bg_movie_np = np.array(bg_movie)
    print("Background movie generation complete (sampled).")
    return bg_movie_np, sample_paths


def foreground_movie(bg_movie, sample_paths, threshold=30):
    """ Part 6: Create foreground movie """
    fg_movie = []

    for i in range(len(bg_movie)):
        img = cv2.imread(sample_paths[i])
        bg = bg_movie[i]

        # 计算差分
        diff = cv2.absdiff(img, bg)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # 二值化提取 Mask
        _, mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)

        # 形态学操作清除噪点
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 提取前景
        fg_frame = cv2.bitwise_and(img, img, mask=mask)
        fg_movie.append(fg_frame)

    print("Foreground movie generation complete.")
    return np.array(fg_movie)


def main():
    img_dir = os.path.join("resources", "images", "frames")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    if not img_paths:
        print("错误：未在 resources/images/frames/ 目录下找到图片。请确保数据已就绪。")
        return

    print(f"Total frames found: {len(img_paths)}")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    projectedWidth = 1600
    projectedHeight = 500
    out_size = (projectedWidth, projectedHeight)
    Tr = np.array([[1, 0, 660], [0, 1, 120], [0, 0, 1]], dtype=np.float32)

    # 缝合两帧 (第 270 帧和第 450 帧)
    im270 = img_paths[270 - 1]
    im450 = img_paths[450 - 1]
    stitch_two(im270, im450, Tr, out_size, output_dir)

    # 利用 5 个关键帧拼接全景图
    key_indices = [90, 270, 450, 630, 810]
    panorama_five(img_paths, key_indices, Tr, out_size, output_dir)

    # 生成映射视频 (投影至参考帧平面)
    ref_idx = 450
    mapped_video = map_video(img_paths, ref_idx, Tr, out_size)

    # 提取背景全景图
    if len(mapped_video) > 0:
        bg_pano = background_panorama(mapped_video, output_dir)

        # 将背景投射回视频空间生成背景视频
        bg_movie, sample_paths = background_movie(bg_pano, img_paths, ref_idx, Tr)

        # 剥离出前景视频
        if len(bg_movie) > 0:
            fg_movie = foreground_movie(bg_movie, sample_paths)
            bg_vid_path = os.path.join(output_dir, "output_bg.mp4")
            fg_vid_path = os.path.join(output_dir, "output_fg.mp4")
            # 使用 ffmpeg 将 np 数组写成视频文件=
            vidwrite_from_numpy(bg_vid_path, bg_movie)
            vidwrite_from_numpy(fg_vid_path, fg_movie)

if __name__ == '__main__':
    main()