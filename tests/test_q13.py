import numpy as np
import pytest
import source.q13 as q13


def test__get_maxmin_diff():
    in_img = np.zeros((5, 5), np.uint8)
    in_img[:, :] = 100
    in_img[0, 0] = 50
    in_img[1, 0] = 200
    assert q13._get_maxmin_diff(in_img) == 150


def test__get_maxmin_diff_same_num():
    in_img = np.zeros((5, 5), np.uint8)
    in_img[:, :] = 100
    assert q13._get_maxmin_diff(in_img) == 0


def test__get_maxmin_diff_negative_num():
    in_img = np.zeros((5, 5), np.uint8)
    in_img[:, :] = -100
    in_img[0, 0] = -50
    in_img[1, 0] = -200
    assert q13._get_maxmin_diff(in_img) == 150
