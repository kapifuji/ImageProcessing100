import numpy as np
import pytest
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


def test_fix_overflow():
    in_img = np.ones((10, 10, 3), np.float)
    in_img[:, :, 0] = -1
    in_img[:, :, 1] = 300
    out_img = np.ones((10, 10, 3), np.float)
    out_img[:, :, 0] = 0
    out_img[:, :, 1] = 255
    assert (q9.fix_overflow(in_img) == out_img).all()


def test_apply_filter_BGR(mocker):
    in_img = np.zeros((3, 3, 3), np.uint8)
    kernel = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
    ])

    add_pad_img = np.zeros((5, 5, 3), np.uint8)
    add_pad_mock = mocker.Mock(return_value=add_pad_img)
    mocker.patch.object(q9, "add_padding", add_pad_mock)

    f_over_img = np.zeros((3, 3, 3), np.uint8)
    f_over_img[:, :, :] = 5
    f_over_mock = mocker.Mock(return_value=f_over_img)
    mocker.patch.object(q9, "fix_overflow", f_over_mock)

    out_img = np.zeros((3, 3, 3), np.uint8)
    out_img[:, :, :] = 5
    fil_func_mock = mocker.Mock(return_value=5)
    assert (q9.apply_filter(
        in_img, kernel.shape[0], fil_func_mock) == out_img).all()

    assert add_pad_mock.call_count == 1
    assert (add_pad_mock.call_args_list[0][0] == in_img).all()

    fil_func_arg = np.zeros((3, 3), np.uint8)
    assert (fil_func_mock.call_args_list[-1][0][0] == fil_func_arg).all()

    f_over_arg = np.zeros((3, 3, 3), np.float)
    f_over_arg[:, :, :] = 5
    assert f_over_mock.call_count == 1
    assert (f_over_mock.call_args_list[0][0] == f_over_arg).all()


def test_apply_filter_gray(mocker):
    in_img = np.zeros((3, 3), np.uint8)
    kernel = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
    ])

    add_pad_img = np.zeros((5, 5), np.uint8)
    add_pad_mock = mocker.Mock(return_value=add_pad_img)
    mocker.patch.object(q9, "add_padding", add_pad_mock)

    f_over_img = np.zeros((3, 3), np.uint8)
    f_over_img[:, :] = 5
    f_over_mock = mocker.Mock(return_value=f_over_img)
    mocker.patch.object(q9, "fix_overflow", f_over_mock)

    out_img = np.zeros((3, 3), np.uint8)
    out_img[:, :] = 5
    fil_func_mock = mocker.Mock(return_value=5)
    assert (q9.apply_filter(
        in_img, kernel.shape[0], fil_func_mock) == out_img).all()

    assert add_pad_mock.call_count == 1
    assert (add_pad_mock.call_args_list[0][0] == in_img).all()

    fil_func_arg = np.zeros((3, 3), np.uint8)
    assert (fil_func_mock.call_args_list[-1][0][0] == fil_func_arg).all()

    f_over_arg = np.zeros((3, 3), np.float)
    f_over_arg[:, :] = 5
    assert f_over_mock.call_count == 1
    assert (f_over_mock.call_args_list[0][0] == f_over_arg).all()
