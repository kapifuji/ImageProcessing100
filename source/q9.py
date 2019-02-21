import cv2
import numpy as np
import traceback


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
        raise ValueError("カネールは正方行列でなければなりません。")

    try:
        return np.sum(mat * kernel)
    except ValueError:
        traceback.print_exc()
        raise


def apply_filter(bgr_img, kernel):
    if not kernel.shape[0] == kernel.shape[1]:
        raise ValueError
    k_padding = kernel.shape[0] // 2
    out_img = bgr_img.copy()
    for _ in range(k_padding):
        out_img = add_padding(out_img)

    for h in range(k_padding, out_img.shape[0] - k_padding):
        for w in range(k_padding, out_img.shape[1] - k_padding):
            for channel in range(out_img.shape[2]):
                top = h - k_padding
                bottom = h + k_padding + 1
                left = w - k_padding
                right = w + k_padding + 1
                out_img[h, w, channel] = get_filter_value(
                    out_img[top: bottom, left: right, channel], kernel)

    for _ in range(kernel.shape[0] // 2):
        out_img = delete_padding(out_img)

    return out_img


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori_noise.jpg")

    gaussian_kernel = np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16],
    ])

    img = apply_filter(img, gaussian_kernel)

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_9.jpg", img)

if __name__ == "__main__":  # pragma: no cover
    _main()
