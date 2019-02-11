import numpy as np
from source.q1 import BGR2RGB

def test_BGR2RGB():
    inputImg = np.zeros((5, 5, 3), np.uint8)
    inputImg[:, :, 0] = 50
    inputImg[:, :, 1] = 100
    inputImg[:, :, 2] = 150
    outputImg = np.zeros((5, 5, 3), np.uint8)
    outputImg[:, :, 0] = 150
    outputImg[:, :, 1] = 100
    outputImg[:, :, 2] = 50

    assert (BGR2RGB(inputImg) == outputImg).all()