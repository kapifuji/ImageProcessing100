import numpy as np
import pytest
import source.q1 as q1


@pytest.fixture()
def input_img():
    input_img = np.zeros((5, 5, 3), np.uint8)
    input_img[:, :, 0] = 50
    input_img[:, :, 1] = 100
    input_img[:, :, 2] = 150

    return input_img


@pytest.fixture()
def output_img():
    output_img = np.zeros((5, 5, 3), np.uint8)
    output_img[:, :, 0] = 150
    output_img[:, :, 1] = 100
    output_img[:, :, 2] = 50

    return output_img


def test_Conv_BGR2RGB(input_img, output_img):
    assert (q1.conv_BGR2RGB(input_img) == output_img).all()
