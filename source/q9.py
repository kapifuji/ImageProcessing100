import cv2
import numpy as np
import traceback


def add_padding(img, val=0):
    """画像の周囲を指定した値でパディングします。

    Arguments:
        img {numpy.ndarray} -- 画像（3ch）

    Keyword Arguments:
        val {int} -- パディングの値 (default: {0})

    Returns:
        numpy.ndarray -- パディング後の画像（3ch）
    """

    out_img = img.copy()
    out_img = np.insert(out_img, 0, val, 0)
    out_img = np.insert(out_img, -1, val, 0)
    out_img = np.insert(out_img, 0, val, 1)
    out_img = np.insert(out_img, -1, val, 1)

    return out_img


def delete_padding(img):
    """パディングを取り消します。

    Arguments:
        img {numpy.ndarray} -- 画像（3ch）

    Returns:
        numpy.ndarray -- パディング取り消し後の画像（3ch）
    """

    out_img = img.copy()
    out_img = np.delete(out_img, 0, 0)
    out_img = np.delete(out_img, -1, 0)
    out_img = np.delete(out_img, 0, 1)
    out_img = np.delete(out_img, -1, 1)

    return out_img


def get_filter_value(mat, kernel) -> float:
    """行列にフィルタを適用した結果の値を取得します。

    Arguments:
        mat {numpy.ndarray} -- 行列
        kernel {numpy.ndarray} -- カーネル

    Raises:
        ValueError
        -- カーネルが正方行列でないとき
        -- 入力された行列とカーネルの大きさが違うとき

    Returns:
        float -- フィルタ適用結果の値
    """

    if not kernel.shape[0] == kernel.shape[1]:
        raise ValueError("カネールは正方行列でなければなりません。")

    try:
        return np.sum(mat * kernel)
    except ValueError:
        traceback.print_exc()
        raise


def apply_filter(bgr_img, kernel):
    """画像にフィルタを適用します。

    Arguments:
        bgr_img {numpy.ndarray} -- BGR画像（3ch）
        kernel {numpy.ndarray} -- カーネル

    Raises:
        ValueError -- カーネルが正方行列でないとき

    Returns:
        numpy.ndarray -- フィルタ適用後画像（3ch）

    Notes:
        入力はRGB画像でも正常に動作します。
    """

    if not kernel.shape[0] == kernel.shape[1]:
        raise ValueError("カネールは正方行列でなければなりません。")
    k_padding = kernel.shape[0] // 2
    out_img = bgr_img.copy()
    for _ in range(k_padding):
        out_img = add_padding(out_img)

    for h in range(k_padding, out_img.shape[0] - k_padding):
        for w in range(k_padding, out_img.shape[1] - k_padding):
            for channel in range(out_img.shape[2]):
                top = h - k_padding
                bottom = h + k_padding + 1
                left = w - k_padding
                right = w + k_padding + 1
                out_img[h, w, channel] = get_filter_value(
                    out_img[top: bottom, left: right, channel], kernel)

    for _ in range(kernel.shape[0] // 2):
        out_img = delete_padding(out_img)

    return out_img


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori_noise.jpg")

    gaussian_kernel = np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16],
    ])

    img = apply_filter(img, gaussian_kernel)

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_9.jpg", img)

if __name__ == "__main__":  # pragma: no cover
    _main()
