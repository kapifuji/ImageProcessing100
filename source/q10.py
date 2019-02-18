import cv2
import numpy as np
import q9


def apply_median_filter(bgr_img):
    tmp_img = q9.add_padding(bgr_img)
    b = tmp_img[:, :, 0].copy()
    g = tmp_img[:, :, 1].copy()
    r = tmp_img[:, :, 2].copy()

    for h in range(1, tmp_img.shape[0] - 1):
        for w in range(1, tmp_img.shape[1] - 1):
            b[h, w] = np.median(b[h - 1: h + 2, w - 1: w + 2])
            g[h, w] = np.median(g[h - 1: h + 2, w - 1: w + 2])
            r[h, w] = np.median(r[h - 1: h + 2, w - 1: w + 2])

    out_img = np.dstack((b, g, r))

    return q9.delete_padding(out_img)


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori_noise.jpg")

    img = apply_median_filter(img)

    cv2.imshow("result", img)
    cv2.waitKey(0)

    cv2.imwrite(r"img/answer_10.jpg", img)

if __name__ == "__main__":  # pragma: no cover
    _main()
