import numpy as np
import pytest
import source.q3 as q3

@pytest.fixture()
def grayImg():
    grayImg = np.zeros((2, 2), np.uint8)
    grayImg[0, 0] = 50
    grayImg[0, 1] = 50
    grayImg[1, 0] = 150
    grayImg[1, 1] = 150

    return grayImg

@pytest.fixture()
def outputImg_128():
    outputImg = np.zeros((2, 2), np.uint8)
    outputImg[0, 0] = 0
    outputImg[0, 1] = 0
    outputImg[1, 0] = 255
    outputImg[1, 1] = 255

    return outputImg

@pytest.fixture()
def outputImg_51():
    outputImg = np.zeros((2, 2), np.uint8)
    outputImg[0, 0] = 0
    outputImg[0, 1] = 0
    outputImg[1, 0] = 255
    outputImg[1, 1] = 255

    return outputImg

@pytest.fixture()
def outputImg_50():
    outputImg = np.zeros((2, 2), np.uint8)
    outputImg[0, 0] = 255
    outputImg[0, 1] = 255
    outputImg[1, 0] = 255
    outputImg[1, 1] = 255

    return outputImg

@pytest.fixture(autouse=True)
def setup(mocker, grayImg):
    mocker.patch.object(q3, "BGR2Gray", mocker.Mock(return_value=grayImg))

def testBGR2BinaryTh128(outputImg_128):
    assert (q3.BGR2Binary(np.zeros((2, 2), np.uint8), 128) == outputImg_128).all()

def testBGR2BinaryTh50(outputImg_51):
    assert (q3.BGR2Binary(np.zeros((2, 2), np.uint8), 51) == outputImg_51).all()

def testBGR2BinaryTh51(outputImg_50):
    assert (q3.BGR2Binary(np.zeros((2, 2), np.uint8), 50) == outputImg_50).all()