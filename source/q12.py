import cv2
import numpy as np
import functools
import q9


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori.jpg")

    motion_kernel = np.array([
        [1/3, 0, 0],
        [0, 1/3, 0],
        [0, 0, 1/3],
    ])

    img = q9.apply_filter(img, motion_kernel.shape[0], functools.partial(
            q9.get_filter_value, motion_kernel))

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_12.jpg", img)

if __name__ == "__main__":  # pragma: no cover
    _main()
