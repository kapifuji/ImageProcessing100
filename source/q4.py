import cv2
import numpy as np
import q2


def _get_sb2(gray_img, threshold: int) -> float:
    """クラス間分散を返します。

    Arguments:
        gray_img {numpy.ndarray} -- グレー画像（1ch）
        threshold {int} -- しきい値

    Returns:
        float -- クラス間分散
    """

    pixel_num = gray_img.shape[0] * gray_img.shape[1]
    # しきい値で分離
    c0_img = gray_img[gray_img < threshold]
    c1_img = gray_img[gray_img >= threshold]
    # 総画素数に対する割合
    w0 = np.sum(c0_img >= 0) / pixel_num
    w1 = np.sum(c1_img >= 0) / pixel_num
    # 画素値の平均
    m0 = c0_img.mean() if not len(c0_img) == 0 else 0.0
    m1 = c1_img.mean() if not len(c1_img) == 0 else 0.0
    # クラス間分散 Sb^2
    return w0 * w1 * (m0 - m1)**2


def _get_otsu_threshold(gray_img) -> int:
    """大津の二値化によるしきい値を返します。

    Arguments:
        gray_img {numpy.ndarray} -- グレー画像（1ch）

    Returns:
        int -- しきい値
    """

    opt_threshold = 0
    max_sb2 = 0
    # w0 * w1 * (M0 - M1) ^2 が最大になるような t が最適なしきい値
    for threshold in range(1, 256):
        sb2 = _get_sb2(gray_img, threshold)
        if max_sb2 < sb2:
            opt_threshold = threshold
            max_sb2 = sb2

    return opt_threshold


def conv_BGR2otsu_binary(img):
    """BGR画像を大津の二値化により2値画像に変換します。

    Arguments:
        img {numpy.ndarray} -- BGR画像（3ch）

    Returns:
        numpy.ndarray -- 2値画像（1ch）
    """

    gray_img = q2.conv_BGR2gray(img)

    threshold = _get_otsu_threshold(gray_img.copy())

    gray_img[gray_img < threshold] = 0
    gray_img[gray_img >= threshold] = 255

    return gray_img


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori.jpg")

    binary_img = conv_BGR2otsu_binary(img)

    cv2.imshow("result", binary_img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_4.jpg", binary_img)

if __name__ == "__main__":  # pragma: no cover
    _main()
