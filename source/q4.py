import cv2
import numpy as np
from q2 import BGR2Gray

def __getOtsuThreshold(grayImg) -> int:
    pixelNum = grayImg.shape[0] * grayImg.shape[1]
    # (threshold, Sb^2)
    optThreshold = 0
    maxSb2 = 0
    # w0 * w1 * (M0 - M1) ^2 が最大になるような t が最適なしきい値
    
    for threshold in range(1, 257):
        # しきい値で分離
        c0Img = grayImg[grayImg < threshold]
        c1Img = grayImg[grayImg >= threshold]
        # 総画素数に対する割合
        w0 = np.sum(c0Img >= 0) / pixelNum
        w1 = np.sum(c1Img >= 0) / pixelNum
        # 画素値の平均
        M0 = c0Img.mean()
        M1 = c1Img.mean()
        # クラス間分散 Sb^2
        Sb2 = w0 * w1 * (M0 - M1)**2

        if maxSb2 < Sb2:
            optThreshold = threshold
            maxSb2 = Sb2

    return optThreshold

def BGR2OtsuBinary(img):
    grayImg = BGR2Gray(img)

    threshold = __getOtsuThreshold(grayImg.copy())

    grayImg[grayImg < threshold] = 0
    grayImg[grayImg >= threshold] = 255
    
    return grayImg


if __name__ == "__main__":
    img = cv2.imread(r"img/imori.jpg")

    binaryImg = BGR2OtsuBinary(img)

    cv2.imshow("result", binaryImg)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_4.jpg", binaryImg)