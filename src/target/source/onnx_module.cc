/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file onnx_module.cc
 * \brief Source code module, only for viewing
 */
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include "codegen_source_base.h"
#include "../../runtime/file_util.h"
#include "../../runtime/meta_data.h"

namespace tvm {
namespace codegen {

using runtime::TVMArgs;
using runtime::TVMRetValue;
using runtime::PackedFunc;

using runtime::GetFileFormat;
using runtime::GetMetaFilePath;
using runtime::FunctionInfo;
using runtime::SaveBinaryToFile;

// Simulator function
class ONNXSourceModuleNodeNode : public runtime::ModuleNode {
 public:
  ONNXSourceModuleNodeNode(std::string code,
                   std::string fmt)
      : code_(code), fmt_(fmt) {}
  
  std::vector<runtime::NDArray> data_entry_;
  std::string current_subgraph_;
  const char* type_key() const {
    return "onnx";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const ObjectPtr<Object>& sptr_to_self) final {
//    LOG(FATAL) << "C Source module cannot execute, to get executable module"
//               << " build TVM with \'" << fmt_ << "\' runtime support";

     std::cout << "in runtime aaa -------------"<<"\n";
      this->current_subgraph_ = name;
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::cout << "in runtime packed 1 -------------"<<"\n";
      for (auto i = 0; i < args.size(); ++i) {
          CHECK(args[i].type_code() == kTVMNDArrayHandle ||
                args[i].type_code() == kTVMDLTensorHandle)
              << "Expect NDArray or DLTensor as inputs"
              << "\n";
          if (args[i].type_code() == kTVMDLTensorHandle) {
            DLTensor* arg = args[i];
            this->data_entry_[i].CopyFrom(arg);
              //std::cout << "in runtime packed arg -------------"<<arg<<"\n";
          } else {
            runtime::NDArray arg = args[i];
            this->data_entry_[i].CopyFrom(arg);
              //std::cout << "in runtime packed arg -------------"<<arg<<"\n";
          }

          // runtime::NDArray arg = args[i];
          //   this->data_entry_[i].CopyFrom(arg);

        }

//       std::string ext_name = "onnxruntime";
//       auto pf = tvm::runtime::Registry::Get(ext_name);
//       CHECK(pf) << "Failed to find the codegen tool for " << ext_name << "\n";
//       std::string path = code_ + current_subgraph_ + fmt_;ex

//       std::cout << "in packed 2 -------------";
//        DLTensor* out = (*pf)(path, 2);
// std::cout << "in packed 3 -------------";

     // *rv = this->data_entry_.back();

             auto out_idx = graph_[this->curr_subgraph_].back().output;
        if (args[args.size() - 1].type_code() == kTVMDLTensorHandle) {
          DLTensor* arg = args[args.size() - 1];
          this->data_entry_[out_idx].CopyTo(arg);
        } else {
          NDArray arg = args[args.size() - 1];
          this->data_entry_[out_idx].CopyTo(arg);
        }
        *rv = data_entry_.back();
        

      });



  }

  std::string GetSource(const std::string& formatx) final {
    return code_;
  }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "onnx") {
      CHECK_NE(code_.length(), 0);
      SaveBinaryToFile(file_name, code_);
    } else {
      CHECK_EQ(fmt, fmt_)
          << "Can only save to format=" << fmt_;
    }
  }

 protected:
  std::string code_;
  std::string fmt_;
};

runtime::Module ONNXSourceModuleNodeCreate(std::string code, std::string fmt) {
  auto n = make_object<ONNXSourceModuleNodeNode>(code, fmt);
  return runtime::Module(n);
}


TVM_REGISTER_GLOBAL("runtime.ONNXModuleCreate")
.set_body_typed(ONNXSourceModuleNodeCreate);
}  // namespace codegen
}  // namespace tvm
