import cv2
import numpy as np
import functools
import q9


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori.jpg", cv2.IMREAD_GRAYSCALE)

    diff_kernel_v = np.array([
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])

    diff_kernel_h = np.array([
        [0, 0, 0],
        [-1, 1, 0],
        [0, 0, 0],
    ])

    img_v = q9.apply_filter(img, diff_kernel_v.shape[0], functools.partial(
            q9.get_filter_value, kernel=diff_kernel_v))

    img_h = q9.apply_filter(img, diff_kernel_h.shape[0], functools.partial(
            q9.get_filter_value, kernel=diff_kernel_h))

    cv2.imshow("result_v", img_v)
    cv2.waitKey(0)
    cv2.imshow("result_h", img_h)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_14_v.jpg", img_v)
    cv2.imwrite(r"img/answer_14_h.jpg", img_h)

if __name__ == "__main__":  # pragma: no cover
    _main()
