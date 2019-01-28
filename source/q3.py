import cv2
import numpy as np

def BGR2gray(img):
    r = img[:, :, 2].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 0].copy()

    return np.array(0.2126 * r + 0.7152 * g + 0.0722 * b, dtype="uint8")

def BGR2binary(img, Threshold):
    grayImg = BGR2gray(img)

    grayImg[grayImg < Threshold] = 0
    grayImg[grayImg >= Threshold] = 255
    
    return grayImg


if __name__ == "__main__":
    img = cv2.imread(r"img/imori.jpg")

    binaryImg = BGR2binary(img, 128)

    cv2.imshow("result", binaryImg)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_3.jpg", binaryImg)