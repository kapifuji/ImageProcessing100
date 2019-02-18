import cv2
import numpy as np


def add_padding(img, val=0):
    out_img = img.copy()
    out_img = np.insert(out_img, 0, val, 0)
    out_img = np.insert(out_img, -1, val, 0)
    out_img = np.insert(out_img, 0, val, 1)
    out_img = np.insert(out_img, -1, val, 1)

    return out_img


def delete_padding(img):
    out_img = img.copy()
    out_img = np.delete(out_img, 0, 0)
    out_img = np.delete(out_img, -1, 0)
    out_img = np.delete(out_img, 0, 1)
    out_img = np.delete(out_img, -1, 1)

    return out_img


def get_filter_value(mat, kernel):
    if not mat.shape == kernel.shape:
        raise ValueError

    return np.sum(mat * kernel)


def apply_gaussian_filter(bgr_img):
    kernel = np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16],
    ])
    tmp_img = add_padding(bgr_img)
    b = tmp_img[:, :, 0].copy()
    g = tmp_img[:, :, 1].copy()
    r = tmp_img[:, :, 2].copy()

    for h in range(1, tmp_img.shape[0] - 1):
        for w in range(1, tmp_img.shape[1] - 1):
            b[h, w] = get_filter_value(b[h - 1: h + 2, w - 1: w + 2], kernel)
            g[h, w] = get_filter_value(g[h - 1: h + 2, w - 1: w + 2], kernel)
            r[h, w] = get_filter_value(r[h - 1: h + 2, w - 1: w + 2], kernel)

    out_img = np.dstack((b, g, r))

    return delete_padding(out_img)


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori_noise.jpg")

    img = apply_gaussian_filter(img)

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_9.jpg", img)

if __name__ == "__main__":  # pragma: no cover
    _main()
