import cv2
import numpy as np


def _add_padding(img, val=0):
    out_img = img.copy()
    out_img = np.insert(out_img, 0, val, 0)
    out_img = np.insert(out_img, -1, val, 0)
    out_img = np.insert(out_img, 0, val, 1)
    out_img = np.insert(out_img, -1, val, 1)

    return out_img


def _delete_padding(img):
    out_img = img.copy()
    out_img = np.delete(out_img, 0, 0)
    out_img = np.delete(out_img, -1, 0)
    out_img = np.delete(out_img, 0, 1)
    out_img = np.delete(out_img, -1, 1)

    return out_img


def _get_gaussian_value(mat):
    kernel = [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ]

    return np.sum(kernel * mat) / 16


def apply_gaussian_filter(bgr_img):
    b = _add_padding(bgr_img[:, :, 0].copy())
    g = _add_padding(bgr_img[:, :, 1].copy())
    r = _add_padding(bgr_img[:, :, 2].copy())

    for h in range(1, bgr_img.shape[0] - 1):
        for w in range(1, bgr_img.shape[1] - 1):
            b[h, w] = _get_gaussian_value(b[h - 1: h + 2, w - 1: w + 2])
            g[h, w] = _get_gaussian_value(g[h - 1: h + 2, w - 1: w + 2])
            r[h, w] = _get_gaussian_value(r[h - 1: h + 2, w - 1: w + 2])

    out_img = np.dstack((_delete_padding(b), _delete_padding(g), _delete_padding(r)))

    return out_img


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori_noise.jpg")

    img = apply_gaussian_filter(img)

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_9.jpg", img)

if __name__ == "__main__":  # pragma: no cover
    _main()
