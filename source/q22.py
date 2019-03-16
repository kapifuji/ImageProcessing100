import cv2
import numpy as np
import matplotlib.pyplot as plt


def control_histogram(img, mean: float, sd: float):
    """画像のヒストグラムを操作します

    Arguments:
        img {numpy.ndarray} -- 画像
        mean {float} -- ヒストグラムの平均値
        sd {float} -- ヒストグラムの標準偏差

    Returns:
        numpy.ndarray -- 操作後画像
    """

    out_img = img.copy()

    out_img = sd / np.std(out_img) * (out_img - np.mean(out_img)) + mean
    out_img[out_img < 0] = 0
    out_img[out_img > 255] = 255

    return out_img.astype(np.uint8)


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori_dark.jpg")

    img = control_histogram(img, 128, 52)

    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.imwrite(r"img/answer_22_1.jpg", img)

    plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.savefig(r"img/answer_22_2.png")
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    _main()
