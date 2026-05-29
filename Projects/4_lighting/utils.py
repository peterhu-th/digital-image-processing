# -*- coding: utf-8 -*-
import numpy as np
import math
import cv2
import scipy.sparse
import scipy.sparse.linalg
from scipy.interpolate import griddata


# ==========================================
# IO Helpers
# ==========================================
def write_image(image: np.ndarray, image_path: str):
    bgr_image = (image[:, :, [2, 1, 0]] * 255).astype(np.uint8)
    cv2.imwrite(image_path, bgr_image)


def read_image(image_path: str) -> np.ndarray:
    bgr_image = cv2.imread(image_path)
    rgb_image = bgr_image[:, :, [2, 1, 0]]
    return rgb_image.astype(np.float32) / 255


def read_hdr_image(image_path: str) -> np.ndarray:
    bgr_hdr_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    return bgr_hdr_image[:, :, [2, 1, 0]].astype(np.float32)


def write_hdr_image(hdr_image: np.ndarray, image_path: str):
    assert (image_path.endswith('.hdr'))
    rgb_hdr_image = hdr_image[:, :, [2, 1, 0]]
    cv2.imwrite(image_path, rgb_hdr_image)


# ==========================================
# Display Helpers
# ==========================================
def rescale_images_linear(le):
    le_min = le[le != -float('inf')].min()
    le_max = le[le != float('inf')].max()
    le[le == float('inf')] = le_max
    le[le == -float('inf')] = le_min
    return (le - le_min) / (le_max - le_min)


# ==========================================
# HDR Helpers
# ==========================================
def gsolve(Z: np.ndarray, B: np.ndarray, l: int, w):
    N, P = Z.shape
    n = 256
    A = scipy.sparse.lil_matrix(((N * P) + n + 1, n + P), dtype='double')
    b = np.zeros((A.shape[0], 1), dtype='double')

    k = 0
    for i in range(N):
        for j in range(P):
            wij = w[Z[i, j]]
            A[k, Z[i, j]] = wij
            A[k, n + j] = -wij
            b[k, 0] = wij * B[i]
            k += 1

    A[k, 128] = 1
    k += 1

    for i in range(n - 2):
        A[k, i] = l
        A[k, i + 1] = -2 * l
        A[k, i + 2] = l
        k += 1

    x = scipy.sparse.linalg.lsqr(A.tocsr(), b)
    g = x[0][:n]
    lE = x[0][n:]
    return g, lE


def get_equirectangular_image(reflection_vector, hdr_image):
    H, W, C = hdr_image.shape
    rv_x, rv_y, rv_z = np.split(reflection_vector, 3, axis=2)

    theta_ball = math.pi - np.arccos(np.clip(rv_y, -1.0, 1.0))
    phi_ball = np.arctan2(rv_z, rv_x)
    phi_ball[phi_ball != phi_ball] = 0
    phi_ball += 3 * math.pi / 2
    phi_ball %= 2 * math.pi

    EH, EW = 360, 720
    phi_1st_half = np.arange(math.pi, 2 * math.pi, math.pi / (EW // 2))
    phi_2nd_half = np.arange(0, math.pi, math.pi / (EW // 2))

    theta_range = np.arange(0, math.pi, math.pi / EH)
    phi_ranges = np.concatenate((phi_1st_half, phi_2nd_half))
    phis, thetas = np.meshgrid(phi_ranges, theta_range)

    spherical_coord = np.concatenate((phi_ball, theta_ball), axis=2).reshape(-1, 2)
    spherical_vals = hdr_image.reshape(-1, 3)
    equirectangular_coord = np.stack((phis, thetas), axis=2).reshape(-1, 2)

    equirectangular_intensities = []
    for c in range(C):
        equirectangular_intensity = griddata(spherical_coord, spherical_vals[:, c], equirectangular_coord)
        equirectangular_intensities.append(equirectangular_intensity.reshape(EH, EW))

    equirectangular_image = np.stack(equirectangular_intensities, axis=2)
    equirectangular_image[np.isnan(equirectangular_image)] = np.nanmean(equirectangular_image)
    return equirectangular_image.astype(np.float32)