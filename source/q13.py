import cv2
import numpy as np
import q9


def _get_maxmin_diff(mat) -> float:
    """行列要素の最大値と最小値の差を取得します。

    Arguments:
        mat {numpy.ndarray} -- 行列

    Returns:
        float -- 最大値と最小値の差
    """

    return np.max(mat) - np.min(mat)


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori.jpg", 0)

    img = q9.apply_filter(img, 3, _get_maxmin_diff)

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_13.jpg", img)

if __name__ == "__main__":  # pragma: no cover
    _main()
