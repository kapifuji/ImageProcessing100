import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "source"))
import source.q4 as q4


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


@pytest.fixture()
def setup(mocker, gray_img):
    mocker.patch.object(q4.q2, "conv_BGR2gray", mocker.Mock(return_value=gray_img))


def test__get_sb2_th200(gray_img):
    assert q4._get_sb2(gray_img, 200) == 0


def test__get_sb2_th151(gray_img):
    assert q4._get_sb2(gray_img, 151) == 0


def test__get_sb2_th150(gray_img):
    assert q4._get_sb2(gray_img, 150) == 2500


def test__get_sb2_th51(gray_img):
    assert q4._get_sb2(gray_img, 51) == 2500


def test__get_sb2_th50(gray_img):
    assert q4._get_sb2(gray_img, 50) == 0


def test__get_sb2_th20(gray_img):
    assert q4._get_sb2(gray_img, 20) == 0


def test__get_otsu_threshold_notif(mocker, gray_img):
    mocker.patch.object(q4, "_get_sb2", mocker.Mock(return_value=0))
    assert q4._get_otsu_threshold(gray_img) == 0


def test__get_otsu_threshold_if(mocker, gray_img):
    mocker.patch.object(q4, "_get_sb2", mocker.Mock(return_value=100))
    assert q4._get_otsu_threshold(gray_img) == 1


def test_BGR2binary_th128(mocker, setup, output_img_128):
    mocker.patch.object(q4, "_get_otsu_threshold", mocker.Mock(return_value=128))
    assert (q4.conv_BGR2otsu_binary([0]) == output_img_128).all()


def test_BGR2binary_th51(mocker, setup, output_img_51):
    mocker.patch.object(q4, "_get_otsu_threshold", mocker.Mock(return_value=51))
    assert (q4.conv_BGR2otsu_binary([0]) == output_img_51).all()


def test_BGR2binary_th50(mocker, setup, output_img_50):
    mocker.patch.object(q4, "_get_otsu_threshold", mocker.Mock(return_value=50))
    assert (q4.conv_BGR2otsu_binary([0]) == output_img_50).all()
