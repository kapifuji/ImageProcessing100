import numpy as np
import pytest
import source.q6 as q6


@pytest.fixture()
def in_img():
    in_img = np.zeros((2, 2, 3), np.uint8)
    in_img[0, 0, 0] = 0
    in_img[0, 1, 0] = 62
    in_img[1, 0, 0] = 63
    in_img[1, 1, 0] = 126
    in_img[0, 0, 1] = 127
    in_img[0, 1, 1] = 190
    in_img[1, 0, 1] = 191
    in_img[1, 1, 1] = 255

    return in_img


@pytest.fixture()
def out_img():
    out_img = np.zeros((2, 2, 3), np.uint8)
    out_img[:, :, :] = 32
    out_img[0, 0, 0] = 32
    out_img[0, 1, 0] = 32
    out_img[1, 0, 0] = 96
    out_img[1, 1, 0] = 96
    out_img[0, 0, 1] = 160
    out_img[0, 1, 1] = 160
    out_img[1, 0, 1] = 224
    out_img[1, 1, 1] = 224

    return out_img


def test_apply_color_reduction_border(in_img, out_img):
    assert (q6.apply_color_reduction(in_img) == out_img).all()
