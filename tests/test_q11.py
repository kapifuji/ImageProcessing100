import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "source"))
import source.q11 as q11


def test_apply_smoothing_filter(mocker):
    in_img = np.zeros((3, 3, 3), np.uint8)

    add_pad_img = np.zeros((5, 5, 3), np.uint8)
    add_pad_mock = mocker.Mock(return_value=add_pad_img)
    mocker.patch.object(q11.q9, "add_padding", add_pad_mock)

    get_fil_mock = mocker.Mock(return_value=1)
    mocker.patch.object(q11.np, "mean", get_fil_mock)

    del_pad_img = np.zeros((3, 3, 3), np.uint8)
    del_pad_img[:, :, :] = 1
    del_pad_mock = mocker.Mock(return_value=del_pad_img)    
    mocker.patch.object(q11.q9, "delete_padding", del_pad_mock)

    out_img = np.zeros((3, 3, 3), np.uint8)
    out_img[:, :, :] = 1
    assert (q11.apply_smoothing_filter(in_img, 3) == out_img).all()

    assert add_pad_mock.call_count == 1
    assert (add_pad_mock.call_args_list[0][0] == in_img).all()

    del_pad_arg = np.zeros((5, 5, 3), np.uint8)
    del_pad_arg[1:4, 1:4, :] = 1
    assert del_pad_mock.call_count == 1
    assert (del_pad_mock.call_args_list[0][0] == del_pad_arg).all()
