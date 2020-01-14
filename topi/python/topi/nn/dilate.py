# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, redefined-builtin
"""Dilation operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import util
from .. import tag
from .pad import pad
from .util import get_pad_tuple
from ..util import simplify


@tvm.tag_scope(tag=tag.INJECTIVE+",dilate")
def dilate(data, strides, name="DilatedInput"):
    """Dilate data with zeros.

    Parameters
    ----------
    data : tvm.Tensor
        n-D, can be any layout.

    strides : list / tuple of n ints
        Dilation stride on each dimension, 1 means no dilation.

    name : str, optional
        The name prefix operators generated

    Returns
    -------
    Output : tvm.Tensor
        n-D, the same layout as data.
    """
    n = len(data.shape)
    if len(strides) != n:
        raise ValueError("data dimension and strides size dismatch : %d vs %d" % (
            n, len(strides)))

    out_shape = tuple(
        tvm.ir_pass.Simplify((data.shape[i] - 1) * strides[i] + 1) for i in range(n))

    def _dilate(*indices):
        not_zero = []
        index_tuple = []
        idxdiv = tvm.indexdiv
        idxmod = tvm.indexmod
        for i in range(n):
            if not util.equal_const_int(strides[i], 1):
                index_tuple.append(idxdiv(indices[i], strides[i]))
                not_zero.append(idxmod(indices[i], strides[i]).equal(0))
            else:
                index_tuple.append(indices[i])
        if not_zero:
            not_zero = tvm.all(*not_zero)
            return tvm.if_then_else(not_zero, data(*index_tuple), tvm.const(0.0, data.dtype))
        return data(*index_tuple)

    return tvm.compute(out_shape, _dilate, name=name)

@tvm.target.generic_func
def dilation2d(input, filter, strides, padding, dilation, layout='NCHW', out_dtype=None):
    """Dilation2D operator.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    filter : tvm.Tensor
        3-D with shape [in_channel, filter_height, filter_width]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        layout of data

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    # search platform specific declaration first
    # default declaration
    if layout == 'NCHW':
        return dilation2d_nchw(input, filter, strides, padding, dilation, out_dtype)

    raise ValueError("not support this layout {} yet".format(layout))


def dilation2d_nchw(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Dilation2S operator in NCHW layout.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.Tensor
        3-D with shape [ in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = Input.shape
    channel, kernel_h, kernel_w = Filter.shape
    assert in_channel.value == channel.value, \
        "For Dilation2D input and filter channels should be same."

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))

    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = pad(Input, pad_before, pad_after, name="pad_temp")
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')

    return tvm.compute(
        (batch, in_channel, out_height, out_width),
        lambda nn, ff, yy, xx: tvm.max(
            temp[nn, ff, yy * stride_h + ry * dilation_h,
                 xx * stride_w + rx * dilation_w].astype(out_dtype) +
            Filter[ff, ry, rx].astype(out_dtype),
            axis=[ry, rx]), tag="dilation2d_nchw")
