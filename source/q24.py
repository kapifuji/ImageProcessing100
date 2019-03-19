import cv2
import numpy as np


def apply_gamma_correction(img, gamma: float = 2.2, c: float = 1.0):
    """画像にガンマ補正を適用します。

    Arguments:
        img {numpy.ndarray} -- 元画像

    Keyword Arguments:
        gamma {float} -- ガンマ値 (default: {2.2})
        c {float} -- 定数 (default: {1.0})

    Returns:
        numpy.ndarray -- 適用後画像
    """

    out_img = img.copy().astype(np.float)

    out_img /= 255

    out_img = (1 / c * out_img) ** (1 / gamma)

    out_img *= 255

    return out_img.astype(np.uint8)


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori_gamma.jpg")

    img = apply_gamma_correction(img)

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_24.jpg", img)

if __name__ == "__main__":  # pragma: no cover
    _main()
