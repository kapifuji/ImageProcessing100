import cv2
import numpy as np
import matplotlib.pyplot as plt


def _main():  # pragma: no cover
    img = cv2.imread(r"img/imori_dark.jpg")

    plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.savefig(r"img/answer_20.png")
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    _main()
