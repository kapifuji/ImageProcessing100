import cv2
import numpy as np
import functools
import q9


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori.jpg", 0)

    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ])

    img = q9.apply_filter(img, laplacian_kernel.shape[0], functools.partial(
            q9.get_filter_value, kernel=laplacian_kernel))

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_17.jpg", img)

if __name__ == "__main__":  # pragma: no cover
    _main()
