import tvm._ffi
import os
from tvm.relay.converter import to_onnx
import shutil


def create_runtime_module(model_dir, fmt):
    runtime_func = "runtime.CSourceModuleCreate"
    fcreate = tvm._ffi.get_global_func(runtime_func)
    return fcreate(model_dir, fmt)


@tvm._ffi.register_func("target.build.onnx")
def onnx_compiler(ref, a):
    model_dir = os.getcwd()
    if isinstance(ref, tvm.ir.module.IRModule):
        model_path = "{}/{}.onnx".format(model_dir, "asd")
        # if os.path.exists(model_path):
        #     os.remove(model_path)
        a= to_onnx(ref, {}, "asd", path=model_path)

    return create_runtime_module(a, 'onnx')

@tvm._ffi.register_func("relay.ext.onnx")
def coreml_compiler(ref):
    model_dir = os.getcwd()
    if isinstance(ref, tvm.ir.module.IRModule):
        for var, func in ref.functions.items():
            name = var.name_hint
            model_path = "{}/{}.onnx".format(model_dir, name)
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            to_onnx(ref, {}, name, path=model_path)

    return create_runtime_module(model_dir, 'onnx')
