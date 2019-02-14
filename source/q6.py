import cv2
import numpy as np


def apply_color_reduction(bgr_img):
    out_img = bgr_img.copy()
    out_img[(0 <= out_img) & (out_img < 63)] = 32
    out_img[(63 <= out_img) & (out_img < 127)] = 96
    out_img[(127 <= out_img) & (out_img < 191)] = 160
    out_img[(191 <= out_img) & (out_img < 256)] = 224

    return out_img

if __name__ == "__main__":
    img = cv2.imread(r"img/imori.jpg")

    img = apply_color_reduction(img)

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_6.jpg", img)
