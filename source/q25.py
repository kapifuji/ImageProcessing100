import cv2
import numpy as np


def scale_nearest_eighbor(img, rate: float):
    """最近傍補間による拡大縮小

    Arguments:
        img {numpy.ndarray} -- 元画像
        rate {float} -- 拡大縮小率

    Returns:
        [numpy.ndarray] -- 拡大縮小後画像
    """

    h = img.shape[0]
    w = img.shape[1]

    ex_h = int(h * rate)
    ex_w = int(w * rate)

    ex_x = np.tile(np.arange(ex_w), (ex_h, 1))
    ex_y = np.arange(ex_h).repeat(ex_w).reshape(ex_h, ex_w)

    ex_x = np.minimum(np.round(ex_x / rate), w - 1).astype(np.int)
    ex_y = np.minimum(np.round(ex_y / rate), h - 1).astype(np.int)

    out_img = img[ex_y, ex_x]

    return out_img


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori.jpg")

    img = scale_nearest_eighbor(img, 1.5)

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_25.jpg", img)

if __name__ == "__main__":  # pragma: no cover
    _main()
