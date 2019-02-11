import cv2
import numpy as np

def BGR2Gray(img):
    r = img[:, :, 2].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 0].copy()

    return np.array(0.2126 * r + 0.7152 * g + 0.0722 * b, dtype="uint8")

if __name__ == "__main__":
    img = cv2.imread(r"img/imori.jpg")

    grayImg = BGR2Gray(img)

    cv2.imshow("result", grayImg)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_2.jpg", grayImg)