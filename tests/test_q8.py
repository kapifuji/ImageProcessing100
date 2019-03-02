import numpy as np
import pytest
import source.q8 as q8


@pytest.fixture()
def in_img():
    in_img = np.zeros((24, 24, 3), np.uint8)
    in_img[0:4:2, 0:4:2, ::] = 60
    in_img[4:8:2, 4:8:2, ::] = 20
    in_img[16:18:2, 16:18:2, 2] = 55
    in_img[18:24:2, 18:24:2, 2] = 150

    return in_img


@pytest.fixture()
def out_img():
    out_img = np.zeros((24, 24, 3), np.uint8)
    out_img[0:8, 0:8, ::] = 60
    out_img[16:24, 16:24, 2] = 150

    return out_img


def test_apply_max_pooling_8x8(in_img, out_img):
    assert (q8.apply_max_pooling_8x8(in_img) == out_img).all()
