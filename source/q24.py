import cv2
import numpy as np


def apply_gamma_correction(img, c: float, gamma: float = 2.2):
    out_img = img.copy().astype(np.float)

    out_img /= 255

    out_img = (1 / c * out_img) ** (1 / gamma)

    out_img *= 255

    return out_img.astype(np.uint8)


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori_gamma.jpg")

    img = apply_gamma_correction(img, 1)

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_24.jpg", img)

if __name__ == "__main__":  # pragma: no cover
    _main()
