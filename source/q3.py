import cv2
import numpy as np
from q2 import BGR2Gray

def BGR2Binary(img, threshold):
    grayImg = BGR2Gray(img)

    grayImg[grayImg < threshold] = 0
    grayImg[grayImg >= threshold] = 255
    
    return grayImg


if __name__ == "__main__":
    img = cv2.imread(r"img/imori.jpg")

    binaryImg = BGR2Binary(img, 128)

    cv2.imshow("result", binaryImg)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_3.jpg", binaryImg)