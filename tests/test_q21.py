import numpy as np
import pytest
import source.q21 as q21


def test_apply_gscale_transform():
    in_img = np.zeros((5, 5), np.uint8)
    in_img[:, :] = 100
    in_img[0, 0] = 50
    in_img[0, 1] = 150
    out_img = np.zeros((5, 5), np.uint8)
    out_img[:, :] = 125
    out_img[0, 0] = 0
    out_img[0, 1] = 250
    assert (q21.apply_gscale_transform(in_img, 0, 250) == out_img).all()
