import numpy as np
import pytest
import source.q1 as q1

@pytest.fixture()
def inputImg():
    inputImg = np.zeros((5, 5, 3), np.uint8)
    inputImg[:, :, 0] = 50
    inputImg[:, :, 1] = 100
    inputImg[:, :, 2] = 150

    return inputImg

@pytest.fixture()
def outputImg():
    outputImg = np.zeros((5, 5, 3), np.uint8)
    outputImg[:, :, 0] = 150
    outputImg[:, :, 1] = 100
    outputImg[:, :, 2] = 50

    return outputImg

def testBGR2RGB(inputImg, outputImg):
    assert (q1.BGR2RGB(inputImg) == outputImg).all()