import cv2
import numpy as np
import matplotlib.pyplot as plt


def flatten_histogram(img):
    """画像のヒストグラムを平坦化します

    Arguments:
        img {numpy.ndarray} -- 画像

    Returns:
        numpy.ndarray -- 平坦化画像
    """

    out_img = img.copy()

    z_max = 255
    pix_sum = img.shape[0] * img.shape[1] if img.ndim == 2 \
        else img.shape[0] * img.shape[1] * img.shape[2]
    h_sum = 0

    for i in range(0, 256):
        index = np.where(img == i)
        h_sum += len(img[index])
        out_img[index] = z_max / pix_sum * h_sum

    return out_img.astype(np.uint8)


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori.jpg")

    img = flatten_histogram(img)

    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.imwrite(r"img/answer_23_1.jpg", img)

    plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.savefig(r"img/answer_23_2.png")
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    _main()
