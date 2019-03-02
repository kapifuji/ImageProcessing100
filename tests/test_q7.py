import numpy as np
import pytest
import source.q7 as q7


@pytest.fixture()
def in_img():
    in_img = np.zeros((24, 24, 3), np.uint8)
    in_img[0:8:2, 0:8:2, ::] = 60
    in_img[16:24:2, 16:24:2, 2] = 70

    return in_img


@pytest.fixture()
def out_img():
    out_img = np.zeros((24, 24, 3), np.uint8)
    out_img[0:8, 0:8, ::] = 15
    out_img[16:24, 16:24, 2] = 17

    return out_img


def test_apply_mean_pooling_8x8(in_img, out_img):
    assert (q7.apply_mean_pooling_8x8(in_img) == out_img).all()
