import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

# Hybrid Images
def hybridImage(im1, im2, sigma_low, sigma_high):
    kernel_half_low = int(3 * sigma_low)
    gauss_kernel_low = utils.gaussian_kernel(sigma_low, kernel_half_low)
    low_pass = cv2.filter2D(im1, -1, gauss_kernel_low)

    kernel_half_high = int(3 * sigma_high)
    gauss_kernel_high = utils.gaussian_kernel(sigma_high, kernel_half_high)
    low_pass_im2 = cv2.filter2D(im2, -1, gauss_kernel_high)
    high_pass = im2 - low_pass_im2

    hybrid = low_pass + high_pass
    hybrid = np.clip(hybrid, 0.0, 1.0)

    return hybrid, low_pass, high_pass

im1_file = 'pictures/nutmeg.jpg'
im2_file = 'pictures/DerekPicture.jpg'

im1 = cv2.imread(im1_file, cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread(im2_file, cv2.IMREAD_GRAYSCALE)

if im1 is not None and im2 is not None:
    im1 = np.float32(im1) / 255.0
    im2 = np.float32(im2) / 255.0
    im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))
    
    sigma_low = 15
    sigma_high = 10
    hybrid, low_pass, high_pass = hybridImage(im1, im2, sigma_low, sigma_high)
    
    cv2.imwrite('test_hybrid.jpg', (hybrid * 255).astype(np.uint8))
    cv2.imwrite('test_low.jpg', (low_pass * 255).astype(np.uint8))
    # shift high pass to 0-255 range for visibility
    cv2.imwrite('test_high.jpg', ((high_pass + 0.5) * 255).astype(np.uint8))
    print("Done")
else:
    print("Files not found")
