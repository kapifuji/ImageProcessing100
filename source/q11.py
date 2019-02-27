import cv2
import numpy as np
import q9


def apply_smoothing_filter(bgr_img, k_size):
    """画像に平滑化フィルタを適用します。

    Arguments:
        bgr_img {numpy.ndarray} -- BGR画像（3ch）
        kernel {int} -- カーネルサイズ

    Returns:
        numpy.ndarray -- フィルタ適用後画像（3ch）

    Notes:
        入力はRGB画像でも正常に動作します。
    """

    k_padding = k_size // 2
    out_img = bgr_img.copy()
    for _ in range(k_padding):
        out_img = q9.add_padding(out_img)

    for h in range(k_padding, out_img.shape[0] - k_padding):
        for w in range(k_padding, out_img.shape[1] - k_padding):
            for channel in range(out_img.shape[2]):
                top = h - k_padding
                bottom = h + k_padding + 1
                left = w - k_padding
                right = w + k_padding + 1
                out_img[h, w, channel] = np.mean(out_img[top: bottom, left: right, channel])

    for _ in range(k_padding):
        out_img = q9.delete_padding(out_img)

    return out_img


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori.jpg")

    img = apply_smoothing_filter(img, 3)

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_11.jpg", img)

if __name__ == "__main__":  # pragma: no cover
    _main()
