import cv2
import numpy as np


def scale_bi_linear(img, rate: float):
    """bi-linear補間による拡大縮小

    Arguments:
        img {numpy.ndarray} -- 元画像
        rate {float} -- 拡大縮小率

    Returns:
        [numpy.ndarray] -- 拡大縮小後画像
    """

    h = img.shape[0]
    w = img.shape[1]

    out_h = int(h * rate)
    out_w = int(w * rate)

    ex_x = np.tile(np.arange(out_w), (out_h, 1))
    ex_y = np.arange(out_h).repeat(out_w).reshape(out_h, out_w)

    out_x = np.minimum(ex_x / rate, w - 2).astype(np.float)
    out_y = np.minimum(ex_y / rate, h - 2).astype(np.float)

    out_idx_x = np.minimum(np.floor(ex_x / rate), w - 2).astype(np.int)
    out_idx_y = np.minimum(np.floor(ex_y / rate), h - 2).astype(np.int)

    dx = out_x - out_idx_x
    dy = out_y - out_idx_y

    dx = dx[:, :, np.newaxis].repeat(3, -1)
    dy = dy[:, :, np.newaxis].repeat(3, -1)

    # OpenCVは 縦 x 横 で定義されているので、一般的な定義と比べ x と y が逆転している
    out_img = (1 - dy) * (1 - dx) * img[out_idx_y, out_idx_x] + \
        dy * (1 - dx) * img[out_idx_y + 1, out_idx_x] + \
        (1 - dy) * dx * img[out_idx_y, out_idx_x + 1] + \
        dy * dx * img[out_idx_y + 1, out_idx_x + 1]

    out_img[out_img < 0] = 0
    out_img[out_img > 255] = 255

    return out_img.astype(np.uint8)


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori.jpg")

    img = scale_bi_linear(img, 1.5)

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_26.jpg", img)

if __name__ == "__main__":  # pragma: no cover
    _main()
