#!/bin/bash
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

set -e
set -u

export PYTHONPATH=python:topi/python:apps/extension/python
export LD_LIBRARY_PATH="build:${LD_LIBRARY_PATH:-}"

rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc

# Test TVM
make cython3


#Install onnx & onnxruntime for testing Relay to ONNX implementation
echo "Installing onnx and onnx runtime"
pip3 install onnx --user
pip3 install onnxruntime --user


# Test extern package
cd apps/extension
rm -rf lib
make
cd ../..

python3 -m pytest -v apps/extension/tests

TVM_FFI=ctypes python3 -m pytest -v tests/python/integration
TVM_FFI=ctypes python3 -m pytest -v tests/python/contrib

TVM_FFI=ctypes python3 -m pytest -v tests/python/relay

# Do not enable OpenGL
# TVM_FFI=cython python -m pytest -v tests/webgl
# TVM_FFI=ctypes python3 -m pytest -v tests/webgl
