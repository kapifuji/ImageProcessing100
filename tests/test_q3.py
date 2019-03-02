import numpy as np
import pytest
import source.q3 as q3


@pytest.fixture()
def gray_img():
    gray_img = np.zeros((2, 2), np.uint8)
    gray_img[0, 0] = 50
    gray_img[0, 1] = 50
    gray_img[1, 0] = 150
    gray_img[1, 1] = 150

    return gray_img


@pytest.fixture()
def output_img_128():
    output_img = np.zeros((2, 2), np.uint8)
    output_img[0, 0] = 0
    output_img[0, 1] = 0
    output_img[1, 0] = 255
    output_img[1, 1] = 255

    return output_img


@pytest.fixture()
def output_img_51():
    output_img = np.zeros((2, 2), np.uint8)
    output_img[0, 0] = 0
    output_img[0, 1] = 0
    output_img[1, 0] = 255
    output_img[1, 1] = 255

    return output_img


@pytest.fixture()
def output_img_50():
    output_img = np.zeros((2, 2), np.uint8)
    output_img[0, 0] = 255
    output_img[0, 1] = 255
    output_img[1, 0] = 255
    output_img[1, 1] = 255

    return output_img


@pytest.fixture(autouse=True)
def setup(mocker, gray_img):
    mocker.patch.object(q3.q2, "conv_BGR2gray", mocker.Mock(return_value=gray_img))


def test_BGR2binary_th128(output_img_128):
    assert (q3.conv_BGR2binary([0], 128) == output_img_128).all()


def test_BGR2binary_th51(output_img_51):
    assert (q3.conv_BGR2binary([0], 51) == output_img_51).all()


def test_BGR2binary_th50(output_img_50):
    assert (q3.conv_BGR2binary([0], 50) == output_img_50).all()
