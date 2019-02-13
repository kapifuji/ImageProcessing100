import cv2
import numpy as np


def conv_BGR2HSV(bgr_img):
    bgr = bgr_img.astype(np.float32).copy() / 255
    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]
    max_val = np.max(bgr, axis=2)
    min_val = np.min(bgr, axis=2)

    h = np.empty_like(b)
    h[min_val == max_val] = 0
    idx = min_val == b
    h[idx] = 60 * (g[idx] - r[idx]) / (max_val[idx] - min_val[idx]) + 60
    idx = min_val == r
    h[idx] = 60 * (b[idx] - g[idx]) / (max_val[idx] - min_val[idx]) + 180
    idx = min_val == g
    h[idx] = 60 * (r[idx] - b[idx]) / (max_val[idx] - min_val[idx]) + 300

    hsv_img = np.zeros_like(bgr)
    hsv_img[:, :, 0] = h
    hsv_img[:, :, 1] = max_val - min_val
    hsv_img[:, :, 2] = max_val

    return hsv_img


def conv_HSV2BGR(hsv_img):
    V = hsv_img[:, :, 2]
    C = hsv_img[:, :, 1]
    H_p = hsv_img[:, :, 0] / 60
    X = C * (1 - np.abs(H_p % 2 - 1))
    Z = np.zeros_like(C)
    vals = [[Z, X, C], [Z, C, X], [X, C, Z], [C, X, Z], [C, Z, X], [X, Z, C]]

    bgr_img = np.zeros_like(hsv_img)

    for i in range(6):
        idx = (i <= H_p) * (H_p < (i + 1))
        bgr_img[:, :, 0][idx] = (V - C)[idx] + vals[i][0][idx]
        bgr_img[:, :, 1][idx] = (V - C)[idx] + vals[i][1][idx]
        bgr_img[:, :, 2][idx] = (V - C)[idx] + vals[i][2][idx]

    return (bgr_img * 255).astype(np.uint8)


def inv_hue(hsv_img):
    hsv = hsv_img.copy()
    hsv[:, :, 0] = (hsv[:, :, 0] + 180) % 360

    return hsv

if __name__ == "__main__":
    img = cv2.imread(r"img/imori.jpg")

    hsv_img = conv_BGR2HSV(img)
    hsv_img = inv_hue(hsv_img)
    img = conv_HSV2BGR(hsv_img)

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_5.jpg", img)
