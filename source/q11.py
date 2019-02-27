import cv2
import numpy as np
import q9


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori.jpg")

    img = q9.apply_filter(img, 3, np.mean)

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_11.jpg", img)

if __name__ == "__main__":  # pragma: no cover
    _main()
