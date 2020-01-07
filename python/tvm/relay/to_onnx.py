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
# pylint: disable=invalid-name, import-self, len-as-condition, unused-argument, too-many-lines, redefined-builtin
"""Relay to ONNX serialization """

import numpy
import onnx
import onnx.utils
from onnx import numpy_helper
from tvm.autotvm.graph_tuner.utils.traverse_graph import _expr2graph_impl
from tvm.relay.expr import Call, TupleGetItem, Var, Constant, Tuple


def relay_layout_to_storage_order(storage_order):
    """converter of tvm storage order format parameter to onnx storage order"""
    if storage_order not in ('NCHW', 'NHWC'):
        raise Exception("Mode of storage_order must be either 'NCHW' or 'NHWC'")

    return 0 if storage_order == 'NCHW' else 1


class OpConverter(object):
    """ Operator converter Base Class.
    """
    @classmethod
    def convert_attributes(cls, attrs):
        """convert Relay attributes to ONNX attributes.
           The derived classes should implement this method
           if attributes are required by the operator
           otherwise by default no attributes are passed
        """
        return {}

    @classmethod
    def convert(cls, node, model_container, node_list):
        attrs = cls.convert_attributes(node['node'].attrs)
        node = onnx.helper.make_node(cls.__name__,
                                     node['input_names'],
                                     node['output_names'],
                                     **attrs)
        model_container.add_nodes([node])


def rename(op_name):
    """ This method creates dynamic operator of name op_name with empty attributes
    """
    return type(op_name, (OpConverter,), {})


class Reshape(object):
    """ Operator converter for Reshape.
    """

    @classmethod
    def convert(cls, node, model_container, node_list):
        """Converts Relay operator Reshape to ONNX operator.
           Relay operator accepts shape as attribute but ONNX operator
           accepts it as a input.
        """

        shape = numpy.asarray([a.value for a in node['node'].attrs.newshape],
                              dtype=node['node'].attrs.newshape[0].dtype)
        input_name = 'shape{}'.format(node['output_names'][0])
        node = onnx.helper.make_node(cls.__name__, [node['input_names'][0], input_name],
                                     node['output_names'])
        model_container.add_nodes([node])
        input = onnx.helper.make_tensor_value_info(input_name,
                                                   onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape.dtype],
                                                   shape=shape.shape)
        model_container.add_inputs([input])
        shape_tensor = numpy_helper.from_array(shape, input_name)
        model_container.add_initializers([shape_tensor])


class Conv(OpConverter):
    """ Operator converter for Conv.
    """
    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'group': attrs.get_int("groups"),
            'pads': attrs.get_int_tuple("padding") + attrs.get_int_tuple("padding"),
            'strides': attrs.get_int_tuple("strides"),
            'dilations': attrs.get_int_tuple("dilation"),
            'kernel_shape': attrs.get_int_tuple("kernel_size"),
            }


class MaxPool(OpConverter):
    """ Operator converter for MaxPool.
    """
    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'pads': attrs.get_int_tuple("padding") + attrs.get_int_tuple("padding"),
            'strides': attrs.get_int_tuple("strides"),
            'kernel_shape': attrs.get_int_tuple("pool_size"),
            }


class Transpose(OpConverter):
    """ Operator converter for Transpose.
    """
    @classmethod
    def convert_attributes(cls, attrs):
        return {'perm': attrs.get_int_tuple("axes")} if attrs["axes"] else {}


class MatMul(OpConverter):
    """ Operator converter for MatMul.
    """
    @classmethod
    def convert(cls, node, model_container, node_list):
        output_name = 'inter{}'.format(node['output_names'][0])
        transpose_node = onnx.helper.make_node(Transpose.__name__,
                                               [node['input_names'][1]],
                                               [output_name],
                                               **{'perm': (1, 0)})
        model_container.add_nodes([transpose_node])

        inputs = [node['input_names'][0], output_name]
        matmul_node = onnx.helper.make_node(cls.__name__, inputs, node['output_names'])
        model_container.add_nodes([matmul_node])


class Flatten(OpConverter):
    """ Operator converter for Flatten.
    """
    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'axis': 1,
            }


class BatchNormalization(OpConverter):
    """ Operator converter for BatchNormalization.
    """
    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'epsilon': float(attrs.get_str('epsilon')),
            # 'spatial' : 1 #TODO - version based support
            }


class Dropout(OpConverter):
    """ Operator converter for Dropout.
    """
    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'ratio': float(attrs.get_str('rate')),
        }


class AveragePool(MaxPool):
    """ Operator converter for AveragePool.
    """


class Concat(OpConverter):
    """ Operator converter for Concat.
    """
    @classmethod
    def convert_attributes(cls, attrs):
        return {
            'axis': attrs.get_int("axis"),
            }


class BiasAdd(OpConverter):
    """ Operator converter for BiasAdd.
    """

    @classmethod
    def convert(cls, node, model_container, node_list):

        input_node = node_list[node['inputs'][0][0]]
        data_ndim = len(input_node['types'][0].shape)
        axis = node['node'].attrs.get_int("axis")
        if axis < 0:
            axis = axis + data_ndim
        num_newaxis = data_ndim - axis - 1
        if num_newaxis:
            output_name = 'inter{}'.format(node['output_names'][0])
            transpose_node = onnx.helper.make_node('Unsqueeze',
                                                   [node['input_names'][1]],
                                                   [output_name],
                                                   **{'axes': tuple(range(1, num_newaxis+1))})
            model_container.add_nodes([transpose_node])
        else:
            output_name = node['input_names'][1]

        inputs = [node['input_names'][0], output_name]
        matmul_node = onnx.helper.make_node('Add', inputs, node['output_names'])
        model_container.add_nodes([matmul_node])


relay_to_onnx_op_mapping = {
    'reshape': Reshape,
    'conv2d': Conv,
    'add': rename('Add'),
    'relu': rename('Relu'),
    'transpose': Transpose,
    'dense': MatMul,
    'max_pool2d': MaxPool,
    'batch_flatten': Flatten,
    'multiply': rename('Mul'),
    'bias_add': BiasAdd,
    'batch_norm': BatchNormalization,
    'global_avg_pool2d': rename('GlobalAveragePool'),
    'concatenate': Concat,
    'dropout': Dropout,
    'avg_pool2d': AveragePool,
    'divide': rename('Div')
}


class ModelContainer(object):
    """ A container class to hold  different attributes of ONNX model graph
    """

    def __init__(self, name):
        self._name = name
        self._inputs = []
        self._outputs = []
        self._nodes = []
        self._initializers = []

    def add_inputs(self, inputs):
        self._inputs.extend(inputs)

    def add_outputs(self, outputs):
        self._outputs.extend(outputs)

    def add_nodes(self, nodes):
        self._nodes.extend(nodes)

    def add_initializers(self, initializers):
        self._initializers.extend(initializers)

    def make_model(self):
        graph_def = onnx.helper.make_graph(
            self._nodes,
            self._name,
            self._inputs,
            self._outputs,
            self._initializers
        )

        return onnx.helper.make_model(graph_def, producer_name='relay')


class RelayToONNXConverter(object):
    """A helper class converting topologically sorted  Relay Node list to ONNX model

    Parameters
    ----------
    name : str
       name of the model

    node_list : list
        topologically sorted Relay Node entry list
    """

    def __init__(self, name, node_list, params):
        self._name = {}
        self._mc = ModelContainer(name)
        self._node_list = node_list
        self._params = params

    def convert_to_onnx(self):
        """ Loop through topologically sorted list of Relay nodes and generate a ONNX model"""
        for idx, node_entry in enumerate(self._node_list):
            out_idx = idx
            node = node_entry['node']
            if isinstance(node, Call):
                self._add_node(node_entry, idx)
            elif isinstance(node, Var):
                self._add_input(node_entry, idx)
            elif isinstance(node, Constant):
                self._add_constant_input(node_entry, idx)
            elif isinstance(node, (TupleGetItem, Tuple)):
                out_idx = idx - 1  # Need to work on this. No equivalent ONNX operator found yet
            else:
                raise NotImplementedError("Relay Node of type {0} is not "
                                          "implemented yet".format(type(node)))

            if idx == len(self._node_list) - 1:
                self._add_output(self._node_list[out_idx], out_idx)

        model = self._mc.make_model()
        polished_model = onnx.utils.polish_model(model)
        return polished_model

    def _tuple_to_name(self, input):
        """convert tuple of node indexes to string"""
        return 'node_{0}'.format(input[0])

    def _add_node(self, node_entry, idx):
        """Convert Relay operator node to ONNX opeartor and add it to container nodes list"""
        if node_entry['op'] not in relay_to_onnx_op_mapping and not node_entry['op'].startswith('func_'):
            raise NotImplementedError("Currently the operator '{0}' is "
                                      "not supported.".format(node_entry['op']))

        if not node_entry['op'].startswith('func_'):
            converter = relay_to_onnx_op_mapping[node_entry['op']]()
        else:
            converter = OpConverter()
        node_entry['output_names'] = [self._tuple_to_name([idx, 0, 0])]
        node_entry['input_names'] = []
        for input_idx_tuple in node_entry['inputs']:
            if self._node_list[input_idx_tuple[0]]['name']:
                node_entry['input_names'].append(self._node_list[input_idx_tuple[0]]['name'])
            else:
                node_entry['input_names'].append(self._tuple_to_name(input_idx_tuple))

        converter.convert(node_entry, self._mc, self._node_list)

    def _add_params(self, node_entry, idx):
        """Add param value to initializer and name to inputs"""
        param_name = node_entry['name']
        assert param_name in self._params, "The parameter {0} is not present" \
                                           "in params dict provided.".format(param_name)
        value = self._params[param_name]
        numpy_array = value.asnumpy()
        tensor = numpy_helper.from_array(numpy_array, param_name)
        self._mc.add_initializers([tensor])
        dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[numpy_array.dtype]
        input = onnx.helper.make_tensor_value_info(param_name,
                                                   dtype,
                                                   shape=numpy_array.shape)
        self._mc.add_inputs([input])

    def _add_constant_input(self, node_entry, idx):
        """Create named input for constant and add it to container inputs.
        If input is a parameter then add to param
        """
        node = node_entry['node']
        if not node_entry['name']:
            node_entry['name'] = self._tuple_to_name([idx, 0, 0])
        param_name = node_entry['name']
        self._params[param_name] = node.data
        self._add_params(node_entry, idx)

    def _add_input(self, node_entry, idx):
        """Add input node to container inputs. If input is a parameter then add to param"""
        if node_entry['name'] in self._params:
            self._add_params(node_entry, idx)
        else:
            type = node_entry['types'][0]
            dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[numpy.dtype(type.dtype)]
            input = onnx.helper.make_tensor_value_info(node_entry['name'],
                                                       dtype,
                                                       shape=type.concrete_shape)
            self._mc.add_inputs([input])

    def _add_output(self, node_entry, idx):
        """Add output node to container outputs."""

        type = node_entry['types'][0]  # TODO - what if there are multiple outputs
        dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[numpy.dtype(type.dtype)]
        output = onnx.helper.make_tensor_value_info(self._tuple_to_name([idx, 0, 0]),
                                                    dtype,
                                                    shape=type.concrete_shape)
        self._mc.add_outputs([output])


def to_onnx(relay_module, params, name, path=None):
    """Converts a Relay Function Module into an equivalent ONNX and serialises it to the path

    Parameters
    ----------
    relay_module : tvm.relay.Module
             The relay module object

    params : dict
        dict of the parameter names and NDarray values

    path : str
        The path where ONNX model will be saved

    Returns
    -------
    inferred_model : tvm.relay.Module
        The relay module

    """
    node_list = []  # ONNX needs a topologically sorted list of nodes
    node_dict = {}
    _expr2graph_impl(relay_module["main"], [], node_dict, node_list)
    converter = RelayToONNXConverter(name, node_list, params)
    onnx_model = converter.convert_to_onnx()

    if path:
        onnx.save(onnx_model, path)
    return onnx_model
