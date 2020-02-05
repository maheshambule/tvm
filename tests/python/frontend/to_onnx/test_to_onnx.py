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

"""Relay to ONNX serialization test cases"""
import numpy as np
import tvm
from tvm import relay
from tvm.relay import to_onnx
import onnxruntime as rt


def func_to_onnx(func, name):
    mod = relay.Module()
    mod['main'] = func
    onnx_model = to_onnx.to_onnx(mod, {}, name, None)
    return onnx_model.SerializeToString()


def run_onnx(onnx_model, input_data):
    sess = rt.InferenceSession(onnx_model)
    input_names ={}
    for input, data in zip(sess.get_inputs(), input_data):
        input_names[input.name] = data
    output_name = sess.get_outputs()[0].name
    res = sess.run([output_name], input_names)
    return res[0]


def run_relay(func, data_tuple):
    target = 'llvm'
    ctx = tvm.context('llvm', 0)
    intrp = relay.create_executor("graph", ctx=ctx, target=target)
    relay_res = intrp.evaluate(func)(*data_tuple)
    return relay_res.asnumpy()


def test_add():
    dtype = 'float32'
    t1 = relay.TensorType((5, 10, 5))
    t2 = relay.TensorType((5, 10, 5))
    x = relay.var("x", t1, dtype=dtype)
    y = relay.var("y", t2, dtype=dtype)
    z = relay.add(x, y)
    func = relay.Function([x, y], z)

    x_data = np.random.rand(5, 10, 5).astype(dtype)
    y_data = np.random.rand(5, 10, 5).astype(dtype)

    relay_res = run_relay(func, (x_data, y_data))
    onnx_res = run_onnx(func_to_onnx(func, 'add'), [x_data, y_data])

    np.testing.assert_allclose(relay_res, onnx_res)


def test_bias_add():
    for dtype in ['float16', 'float32']:
        xshape = (10, 2, 3, 4)
        bshape = (2,)
        rtol = 1e-2 if dtype is 'float16' else 1e-5
        x = relay.var("x", shape=xshape, dtype=dtype)
        bias = relay.var("bias", dtype=dtype)
        z = relay.nn.bias_add(x, bias)
        func = relay.Function([x, bias], z)

        x_data = np.random.uniform(size=xshape).astype(dtype)
        y_data = np.random.uniform(size=bshape).astype(dtype)

        relay_res = run_relay(func, (x_data, y_data))
        onnx_res = run_onnx(func_to_onnx(func, 'test_bias_add'), [x_data, y_data])

        np.testing.assert_allclose(relay_res, onnx_res, rtol=rtol)


def test_conv2d():
    def verify_conv2d(dtype, out_dtype, scale, dshape, kshape,
                        padding=(1, 1),
                        groups=1,
                        dilation=(1, 1),
                        **attrs):

        x = relay.var("x", shape=dshape, dtype=dtype)
        w = relay.var("w", dtype=dtype)
        y = relay.nn.conv2d(x, w,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            **attrs)
        func = relay.Function([x, w], y)
        data = np.random.uniform(-scale, scale, size=dshape).astype(dtype)
        kernel = np.random.uniform(-scale, scale, size=kshape).astype(dtype)
        relay_res = run_relay(func, (data, kernel))
        onnx_res = run_onnx(func_to_onnx(func, 'test_conv2d'), [data, kernel])

        tvm.testing.assert_allclose(relay_res, onnx_res, rtol=1e-5, atol=1e-5)

    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    verify_conv2d("float32", "float32", 1, dshape, kshape,
                    padding=(1, 1), channels=32, groups=32, kernel_size=(3, 3))

    dshape = (1, 32, 18, 18)
    kshape = (32, 4, 3, 3)
    verify_conv2d("float32", "float32", 1, dshape, kshape,
                    padding=(1, 1), channels=32, groups=8, kernel_size=(3, 3))

    # also group conv2d
    dshape = (1, 32, 18, 18)
    kshape = (64, 1, 3, 3)
    verify_conv2d("float32", "float32", 1, dshape, kshape,
                    padding=(1, 1), channels=64, groups=32, kernel_size=(3, 3))

    # normal conv2d
    dshape = (1, 3, 224, 224)
    kshape = (10, 3, 3, 3)
    verify_conv2d("float32", "float32", 1, dshape, kshape,
                    padding=(1, 1), channels=10, kernel_size=(3, 3))

    dshape = (1, 3, 224, 224)
    kshape = (10, 3, 3, 3)
    verify_conv2d("float32", "float32", 1, dshape, kshape,
                    padding=(2, 2), channels=10, kernel_size=(3, 3))

    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 3, 3)
    verify_conv2d("float32", "float32", 1, dshape, kshape,
                    padding=(1, 1), channels=10, kernel_size=(3, 3), dilation=(3, 3))

    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 2, 2)
    verify_conv2d("float32", "float32", 1, dshape, kshape,
                    padding=(2, 2), channels=10, kernel_size=(2, 2), dilation=(1, 1))

    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 4, 4)
    verify_conv2d("float32", "float32", 1, dshape, kshape,
                    padding=(1, 1), channels=10, kernel_size=(4, 4))

    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 4, 4)
    verify_conv2d("float32", "float32", 1, dshape, kshape,
                  padding=(1, 1), channels=10, kernel_size=(4, 4))


def test_reshape():
    def verify_reshape(shape, newshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.reshape(x, newshape=newshape)

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        relay_res = run_relay(func, (x_data,))
        onnx_res = run_onnx(func_to_onnx(func, 'test_reshape'), [x_data])
        tvm.testing.assert_allclose(relay_res, onnx_res, rtol=1e-5, atol=1e-5)

    verify_reshape((2, 3, 4), tuple(np.array([4, 2, 3], dtype=np.int64)))
    verify_reshape((2, 3, 4), tuple(np.array([2, 0, 0], dtype=np.int64)))
    verify_reshape((2, 3, 4), tuple(np.array([0, -1], dtype=np.int64)))
    verify_reshape((2, 3, 4), tuple(np.array([-1, 0], dtype=np.int64)))


def test_transpose():
    def verify_reshape(shape, newshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.transpose(x, newshape)

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        relay_res = run_relay(func, (x_data,))
        onnx_res = run_onnx(func_to_onnx(func, 'test_transpose'), [x_data])
        tvm.testing.assert_allclose(relay_res, onnx_res, rtol=1e-5, atol=1e-5)

    verify_reshape((1, 2, 3, 4), (0, 2, 3, 1))
    verify_reshape((1, 2, 3, 4), (0, 3, 2, 1))


def test_dense():
    def verify_dense(d_shape, w_shape):
        data = relay.var("data", relay.TensorType(d_shape, "float32"))
        weight = relay.var("weight", relay.TensorType(w_shape, "float32"))

        func = relay.Function([data, weight], relay.nn.dense(data, weight))
        x_data = np.random.uniform(size=d_shape).astype("float32")
        w_data = np.random.uniform(size=w_shape).astype("float32")
        relay_res = run_relay(func, (x_data,w_data))
        onnx_res = run_onnx(func_to_onnx(func, 'test_dense'), [x_data, w_data])
        tvm.testing.assert_allclose(relay_res, onnx_res, rtol=1e-5, atol=1e-5)

    verify_dense((1, 8), (16, 8))
    verify_dense((1, 4), (3, 4))


def test_max_pool():
    def verify_max_pool(x_shape, pool_size, strides, padding, ceil_mode):
        x = relay.var("x", relay.TensorType(x_shape, "float32"))
        y = tvm.relay.nn.max_pool2d(x, pool_size=pool_size, strides=strides, padding=padding,
                                    ceil_mode=ceil_mode)

        func = relay.Function([x], y)
        x_data = np.random.uniform(size=x_shape).astype("float32")
        relay_res = run_relay(func, (x_data,))
        onnx_res = run_onnx(func_to_onnx(func, 'test_max_pool'), [x_data])
        tvm.testing.assert_allclose(relay_res, onnx_res, rtol=1e-5, atol=1e-5)

    verify_max_pool((1, 4, 16, 16), pool_size=(2, 2), strides=(2, 2), padding=(0, 0), ceil_mode=False)


def test_batch_flatten():
    def verify_test_batch_flatten(d_shape):
        data = relay.var("data", relay.TensorType(d_shape, "float32"))
        func = relay.Function([data], relay.nn.batch_flatten(data))
        x_data = np.random.uniform(size=d_shape).astype("float32")
        relay_res = run_relay(func, (x_data,))
        onnx_res = run_onnx(func_to_onnx(func, 'test_batch_flatten'), [x_data])
        tvm.testing.assert_allclose(relay_res, onnx_res, rtol=1e-5, atol=1e-5)

    verify_test_batch_flatten((1, 2, 3, 4))
    verify_test_batch_flatten((1, 8))


def test_bias_add():
    def verify_bias_add():
        data = relay.var("data", relay.TensorType((1, 16), "float32"))
        bias = relay.var("bias", relay.TensorType((16,), "float32"))
        func = relay.Function([data, bias], relay.nn.bias_add(data, bias))

        x_data = np.random.uniform(size=(1, 16)).astype("float32")
        bias = np.random.uniform(size=(16,)).astype("float32")
        relay_res = run_relay(func, (x_data,bias))
        onnx_res = run_onnx(func_to_onnx(func, 'test_bias_add'), [x_data, bias])
        tvm.testing.assert_allclose(relay_res, onnx_res, rtol=1e-5, atol=1e-5)

    verify_bias_add()


def test_batch_norm():
    def verify_batch_norm(axis=1):
        for dtype in ['float16', 'float32']:
            data = relay.var("data", relay.TensorType((2, 4, 4, 1), dtype))
            gamma_shape = (data.type_annotation.shape[axis].value,)
            beta = relay.var("beta", relay.TensorType(gamma_shape, dtype))
            gamma = relay.var("gamma", relay.TensorType(gamma_shape, dtype))
            moving_mean = relay.var("moving_mean", relay.TensorType(gamma_shape, dtype))
            moving_var = relay.var("moving_var", relay.TensorType(gamma_shape, dtype))
            y = relay.nn.batch_norm(data, gamma, beta, moving_mean, moving_var, axis=axis)
            func = relay.Function([data, gamma, beta, moving_mean, moving_var], y[0])

            x_data = np.random.uniform(size=(2, 4, 4, 1)).astype(dtype)
            beta = np.random.uniform(size=gamma_shape).astype(dtype)
            gamma = np.random.uniform(size=gamma_shape).astype(dtype)
            moving_mean = np.random.uniform(size=gamma_shape).astype(dtype)
            moving_var = np.random.uniform(size=gamma_shape).astype(dtype)

            relay_res = run_relay(func, (x_data, gamma, beta, moving_mean, moving_var))
            onnx_res = run_onnx(func_to_onnx(func, 'test_batch_norm'), [x_data, gamma,beta, moving_mean, moving_var])

            tol = 1e-3  # TODO - high tolerance
            tvm.testing.assert_allclose(relay_res, onnx_res, rtol=tol, atol=tol)

    verify_batch_norm(axis=1)
    verify_batch_norm(axis=3)


def test_pad():
    def verify_pad():
        for dtype in ['float16', 'float32']:
            dshape = (4, 10, 7, 7)
            x = relay.var("x", shape=dshape, dtype=dtype)
            y = relay.nn.pad(x, ((1, 1), (2, 2), (3, 3), (4, 4)))
            func = relay.Function([x], y)

            data = np.random.uniform(size=dshape).astype(dtype)
            relay_res = run_relay(func, (data,))
            onnx_res = run_onnx(func_to_onnx(func, 'test_pad'), [data])
            tvm.testing.assert_allclose(relay_res, onnx_res, rtol=1e-5, atol=1e-5)

    verify_pad()


def test_sofmax():
    def verify_sofmax():
        for dtype in ['float32']:

            shape = (10, 4)
            x = relay.var("x", shape=shape, dtype=dtype)
            y = relay.nn.softmax(x, axis=1)
            func = relay.Function([x], y)
            x_data = np.random.uniform(size=shape).astype(dtype)

            relay_res = run_relay(func, (x_data,))
            onnx_res = run_onnx(func_to_onnx(func, 'test_sofmax'), [x_data])
            tvm.testing.assert_allclose(relay_res, onnx_res, rtol=1e-5, atol=1e-5)

    verify_sofmax()


def test_squeeze():
    def verify_squeeze(shape, dtype, axis):
        x = relay.var("x", relay.TensorType(shape, dtype))
        z = relay.squeeze(x, axis=axis)
        func = relay.Function([x], z)

        x_data = np.random.random_sample(shape).astype(dtype)
        relay_res = run_relay(func, (x_data,))
        onnx_res = run_onnx(func_to_onnx(func, 'test_squeeze'), [x_data])
        tvm.testing.assert_allclose(relay_res, onnx_res, rtol=1e-5, atol=1e-5)

    verify_squeeze((1, 3, 2, 5), "float32", None)
    verify_squeeze((1, 3, 1), "float32", [2,])
    verify_squeeze((1, 2, 1, 2, 1), "float32", [0, 2])


def test_mean():
    def verify_mean(data_shape, axis, exclude, keepdims):
        dtype = "float32"
        x = relay.var('x', shape=data_shape, dtype=dtype)
        y = relay.mean(x, axis, keepdims, exclude)
        func = relay.Function([x], y)
        x_data = np.random.uniform(size=data_shape).astype(dtype)
        relay_res = run_relay(func, (x_data,))
        onnx_res = run_onnx(func_to_onnx(func, 'test_mean'), [x_data])
        tvm.testing.assert_allclose(relay_res, onnx_res, rtol=1e-5, atol=1e-5)

    verify_mean((1, 2), 0, False, False)
    verify_mean((1, 2), 0, True, False)
    verify_mean((1, 2), 0, True, True)
    verify_mean((1, 2), 1, True, True)
    verify_mean((3, 2, 1), 1, False, True)


if __name__ == '__main__':
    test_add()
    test_bias_add()
    test_conv2d()
    test_reshape()
    test_transpose()
    test_dense()
    test_max_pool()
    test_batch_flatten()
    test_bias_add()
    test_batch_norm()
    test_pad()
    test_mean()
    test_sofmax()
    test_squeeze()
