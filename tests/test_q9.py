import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "source"))
import source.q9 as q9


def test_add_padding():
    in_img = np.zeros((8, 8, 3), np.uint8)
    out_img = np.zeros((10, 10, 3), np.uint8)
    assert (q9.add_padding(in_img) == out_img).all()


def test_delete_padding():
    in_img = np.zeros((12, 12, 3), np.uint8)
    out_img = np.zeros((10, 10, 3), np.uint8)
    assert (q9.delete_padding(in_img) == out_img).all()


def test_get_filter_value_kernel_is_not_square():
    mat = np.zeros((3, 3), np.uint8)
    kernel = np.zeros((5, 6), np.uint8)
    with pytest.raises(ValueError):
        q9.get_filter_value(mat, kernel)


def test_get_filter_value_matrix_size_is_not_equal():
    mat = np.zeros((3, 3), np.uint8)
    kernel = np.zeros((5, 5), np.uint8)
    with pytest.raises(ValueError):
        q9.get_filter_value(mat, kernel)


def test_get_filter_value():
    mat = np.array([
        [16, 32, 48],
        [16, 32, 48],
        [16, 32, 48],
    ])
    kernel = np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16],
    ])
    assert q9.get_filter_value(mat, kernel) == 32


def test_apply_filter(mocker):
    in_img = np.zeros((3, 3, 3), np.uint8)
    kernel = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
    ])

    add_pad_img = np.zeros((5, 5, 3), np.uint8)
    add_pad_mock = mocker.Mock(return_value=add_pad_img)
    mocker.patch.object(q9, "add_padding", add_pad_mock)

    del_pad_img = np.zeros((3, 3, 3), np.uint8)
    del_pad_img[:, :, :] = 5
    del_pad_mock = mocker.Mock(return_value=del_pad_img)    
    mocker.patch.object(q9, "delete_padding", del_pad_mock)

    out_img = np.zeros((3, 3, 3), np.uint8)
    out_img[:, :, :] = 5
    assert (q9.apply_filter(in_img, kernel.shape[0], mocker.Mock(
            return_value=5)) == out_img).all()

    assert add_pad_mock.call_count == 1
    assert (add_pad_mock.call_args_list[0][0] == in_img).all()

    del_pad_arg = np.zeros((5, 5, 3), np.uint8)
    del_pad_arg[1:4, 1:4, :] = 5
    assert del_pad_mock.call_count == 1
    assert (del_pad_mock.call_args_list[0][0] == del_pad_arg).all()
