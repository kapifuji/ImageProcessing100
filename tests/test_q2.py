import numpy as np
import pytest
import source.q2 as q2

@pytest.fixture()
def inputImg():
    inputImg = np.zeros((5, 5, 3), np.uint8)
    inputImg[:, :, 0] = 100
    inputImg[:, :, 1] = 150
    inputImg[:, :, 2] = 200

    return inputImg

@pytest.fixture()
def outputImg():
    outputImg = np.zeros((5, 5), np.uint8)
    outputImg[:, :] = 0.2126 * 200 + 0.7152 * 150 + 0.0722 * 100

    return outputImg

def testBGR2Gray(inputImg, outputImg):
    assert (q2.BGR2Gray(inputImg) == outputImg).all()