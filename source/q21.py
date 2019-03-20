import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_gscale_transform(img, r_min: int, r_max: int):
    """濃度階調変換を画像へ適用

    Arguments:
        img {numpy.ndarray} -- 画像
        r_min {int} -- レンジ（最小値）
        r_max {int} -- レンジ（最大値）

    Returns:
        numpy.ndarray -- 変換適用後画像
    """

    pre_r_min = img.min()
    pre_r_max = img.max()

    out_img = img.copy()

    out_img[img < pre_r_min] = r_min
    out_img[pre_r_max < img] = r_max
    index = (pre_r_min <= img) & (img <= pre_r_max)
    out_img[index] = (r_max - r_min) / (pre_r_max - pre_r_min) * \
        (img[index] - pre_r_min) + r_min

    return out_img


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori_dark.jpg")

    img = apply_gscale_transform(img, 0, 255)

    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.imwrite(r"img/answer_21_1.jpg", img)

    plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.savefig(r"img/answer_21_2.png")
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    _main()
