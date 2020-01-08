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
# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Dilation2D in python"""


import numpy as np


def dilation2d_python(a_np, w_np, stride, padding):
    """Dilation2D operator in NHWC layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_height, in_width, in_channel]

    w_np : numpy.ndarray
        4-D with shape [filter_height, filter_width, in_channel]

    stride : list of ints in the format
        Stride size, or [1,stride_height, stride_width,1]

    padding : str
        ['VALID', 'SAME']

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    batch, in_channel, in_height, in_width = a_np.shape
    in_channel, kernel_h, kernel_w = w_np.shape

    stride_h = stride[0]
    stride_w = stride[1]


    if padding == 'VALID':
        pad_h = 0
        pad_w = 0
    else:  # 'SAME'
        #pad_h = kernel_h - 1
        pad_h = padding[0]
        #pad_w = kernel_w - 1
        pad_w = padding[1]
    pad_top = int(np.ceil(float(pad_h) / 2))
    pad_bottom = pad_h - pad_top
    pad_left = int(np.ceil(float(pad_w) / 2))
    pad_right = pad_w - pad_left

    # compute the output shape
    out_channel = in_channel
    out_height = (in_height - kernel_h + pad_h) // stride_h + 1
    out_width = (in_width - kernel_w + pad_w) // stride_w + 1

    # change the layout from NHWC to NCHW: required for numpy-appropriate processing
    #at = a_np.transpose((0, 3, 1, 2))
    at=a_np
    #wt = w_np.transpose((2, 0, 1))
    wt=w_np
    bt = np.zeros((batch, out_channel, out_height, out_width))
    # computation
    for n in range(batch):
        for c in range(in_channel):
            if pad_h > 0:
                apad = np.zeros((in_height + pad_h, in_width + pad_w))
                apad[pad_top:-pad_bottom, pad_left:-pad_right] = at[n, c]
            else:
                apad = at[n, c]
            wt_c = wt[c]
            out = []
            for i in range(0,out_height):
                for j in range(0,out_width):
                    s = np.max(apad[i*stride_h:(i*stride_h + kernel_h), j*stride_w:(j*stride_w + kernel_w)] + wt_c)
                    out.append(s)
            out = np.asarray(out).reshape((out_height), (out_width))
            bt[n, c] += out
    return bt