import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "source"))
import source.q2 as q2

@pytest.fixture()
def input_img():
    input_img = np.zeros((5, 5, 3), np.uint8)
    input_img[:, :, 0] = 100
    input_img[:, :, 1] = 150
    input_img[:, :, 2] = 200

    return input_img

@pytest.fixture()
def output_img():
    output_img = np.zeros((5, 5), np.uint8)
    output_img[:, :] = 0.2126 * 200 + 0.7152 * 150 + 0.0722 * 100

    return output_img

def testBGR2Gray(input_img, output_img):
    assert (q2.conv_BGR2gray(input_img) == output_img).all()
