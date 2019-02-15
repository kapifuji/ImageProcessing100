import cv2
import numpy as np


def apply_mean_pooling_8x8(bgr_img):
    out_img = bgr_img.copy()
    for h in range(0, int(out_img.shape[0] / 8)):
        for w in range(0, int(out_img.shape[1] / 8)):
            h_min = h * 8
            w_min = w * 8
            h_max = h * 8 + 8
            w_max = w * 8 + 8
            out_img[h_min: h_max, w_min: w_max, 0] = np.mean(out_img[h_min: h_max, w_min: w_max, 0])
            out_img[h_min: h_max, w_min: w_max, 1] = np.mean(out_img[h_min: h_max, w_min: w_max, 1])
            out_img[h_min: h_max, w_min: w_max, 2] = np.mean(out_img[h_min: h_max, w_min: w_max, 2])

    return out_img

if __name__ == "__main__":
    img = cv2.imread(r"img/imori.jpg")

    img = apply_mean_pooling_8x8(img)

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_7.jpg", img)
