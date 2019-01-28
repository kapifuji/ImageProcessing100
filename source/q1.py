import cv2
import numpy as np

def BGR2RGB(img):
    r = img[:, :, 2].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 0].copy()

    outImg = np.empty_like(img)
    outImg[:, :, 0] = r
    outImg[:, :, 1] = g
    outImg[:, :, 2] = b
    
    return outImg

if __name__ == "__main__":
    img = cv2.imread(r"img/imori.jpg")

    outImg = BGR2RGB(img)    

    cv2.imshow("result", outImg)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_1.jpg", outImg)