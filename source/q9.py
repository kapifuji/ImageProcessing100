import cv2
import numpy as np
import traceback
import functools


def add_padding(img, val=0):
    """画像の周囲を指定した値でパディングします。

    Arguments:
        img {numpy.ndarray} -- 画像

    Keyword Arguments:
        val {int} -- パディングの値 (default: {0})

    Returns:
        numpy.ndarray -- パディング後の画像
    """

    out_img = img.copy()
    out_img = np.insert(out_img, 0, val, 0)
    out_img = np.insert(out_img, -1, val, 0)
    out_img = np.insert(out_img, 0, val, 1)
    out_img = np.insert(out_img, -1, val, 1)

    return out_img


def delete_padding(img):
    """パディングを取り除きます。

    Arguments:
        img {numpy.ndarray} -- 画像

    Returns:
        numpy.ndarray -- パディング取り消し後の画像
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


def fix_overflow(img):
    out_img = img.copy()
    out_img[out_img < 0] = 0
    out_img[out_img > 255] = 255

    return out_img


def apply_filter(img, k_size, fil_func):
    """画像にフィルタを適用します。

    Arguments:
        img {numpy.ndarray} -- BGR画像（3ch）
        k_size {numpy.ndarray} -- カーネルサイズ（3x3なら3）
        fil_func {Callable[[numpy.ndarray], float]} -- フィルタ適用関数

    Returns:
        numpy.ndarray -- フィルタ適用後画像（3ch）

    Notes:
        入力はRGB画像、グレー画像でも正常に動作します。
    """

    k_pad = k_size // 2
    channel_num = 1 if img.ndim == 2 else img.shape[2]
    out_img = np.zeros_like(img, np.float)
    tmp_img = img.copy()

    for _ in range(k_pad):
        tmp_img = add_padding(tmp_img)

    for h in range(0, out_img.shape[0]):
        for w in range(0, out_img.shape[1]):
            for channel in range(channel_num):
                h_end = h + 2 * k_pad + 1
                w_end = w + 2 * k_pad + 1
                if channel_num == 1:
                    out_img[h, w] = fil_func(tmp_img[h: h_end, w: w_end])
                else:
                    out_img[h, w, channel] = fil_func(
                        tmp_img[h: h_end, w: w_end, channel])

    out_img = fix_overflow(out_img)

    return out_img.astype(np.uint8)


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori_noise.jpg")

    gaussian_kernel = np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16],
    ])

    img = apply_filter(img, gaussian_kernel.shape[0], functools.partial(
            get_filter_value, kernel=gaussian_kernel))

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_9.jpg", img)

if __name__ == "__main__":  # pragma: no cover
    _main()
