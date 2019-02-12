import cv2
import numpy as np


def conv_BGR2gray(img):
    r = img[:, :, 2].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 0].copy()

    return np.array(0.2126 * r + 0.7152 * g + 0.0722 * b, dtype="uint8")

if __name__ == "__main__":
    img = cv2.imread(r"img/imori.jpg")

    gray_img = conv_BGR2gray(img)

    cv2.imshow("result", gray_img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_2.jpg", gray_img)
