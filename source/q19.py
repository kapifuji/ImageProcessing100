import cv2
import numpy as np
import functools
import q9


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori_noise.jpg", 0)

    log_kernel = np.array([
        [1, 3, 4, 3, 1],
        [3, 6, 7, 6, 3],
        [4, 7, 8, 7, 4],
        [3, 6, 7, 6, 3],
        [1, 3, 4, 3, 1],
    ]) / 100

    img = q9.apply_filter(img, log_kernel.shape[0], functools.partial(
            q9.get_filter_value, kernel=log_kernel))

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_19.jpg", img)

if __name__ == "__main__":  # pragma: no cover
    _main()
