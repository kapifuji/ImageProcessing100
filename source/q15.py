import cv2
import numpy as np
import functools
import q9


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori.jpg", cv2.IMREAD_GRAYSCALE)

    sobel_kernel_v = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ])

    sobel_kernel_h = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ])

    img_v = q9.apply_filter(img, sobel_kernel_v.shape[0], functools.partial(
            q9.get_filter_value, kernel=sobel_kernel_v))

    img_h = q9.apply_filter(img, sobel_kernel_h.shape[0], functools.partial(
            q9.get_filter_value, kernel=sobel_kernel_h))

    cv2.imshow("result_v", img_v)
    cv2.waitKey(0)
    cv2.imshow("result_h", img_h)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_15_v.jpg", img_v)
    cv2.imwrite(r"img/answer_15_h.jpg", img_h)

if __name__ == "__main__":  # pragma: no cover
    _main()
