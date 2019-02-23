import cv2
import numpy as np


def conv_BGR2RGB(img):
    """BGR画像をRGB画像に変換します。

    Arguments:
        img {numpy.ndarray} -- BGR画像

    Returns:
        [numpy.ndarray] -- RGB画像

    Notes:
        ndarrayのdtypeは入力画像と同じものを返します。
    """

    r = img[:, :, 2].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 0].copy()

    out_img = np.empty_like(img)
    out_img[:, :, 0] = r
    out_img[:, :, 1] = g
    out_img[:, :, 2] = b

    return out_img


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori.jpg")

    out_img = conv_BGR2RGB(img)

    cv2.imshow("result", out_img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_1.jpg", out_img)

if __name__ == "__main__":  # pragma: no cover
    _main()
