/*!
 *  Copyright (c) 2018 by Contributors
 *
 * \brief Subgraph data structure.
 * \file subgraph.h
 */
#ifndef TVM_CONTRIB_SUBGRAPH_SUBGRAPH_H_
#define TVM_CONTRIB_SUBGRAPH_SUBGRAPH_H_

#include <dmlc/json.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace contrib {

/*!
 * \brief Subgraph data structure for the subgraphs executed by other
 * accelerators, such as TensorRT etc.
 * This struct provides utility functions for deserializing
 * a subgraph from the json file. The file is generated by
 * using TVM to compile a model.
 */
struct Subgraph {
  struct Node {
    struct NodeEntry {
      uint32_t node_id;
      uint32_t index;
      uint32_t version;
      void Load(dmlc::JSONReader *reader) {
        reader->BeginArray();
        CHECK(reader->NextArrayItem()) << "invalid json format";
        reader->Read(&node_id);
        CHECK(reader->NextArrayItem()) << "invalid json format";
        reader->Read(&index);
        if (reader->NextArrayItem()) {
          reader->Read(&version);
          CHECK(!reader->NextArrayItem()) << "invalid json format";
        } else {
          version = 0;
        }
      }
    };
    std::string op_name;
    std::string node_name;
    std::unordered_map<std::string, std::string> attrs;
    std::vector<NodeEntry> inputs;
    void Load(dmlc::JSONReader *reader) {
      dmlc::JSONObjectReadHelper helper;
      helper.DeclareField("op", &op_name);
      helper.DeclareField("name", &node_name);
      helper.DeclareField("inputs", &inputs);
      helper.DeclareOptionalField("attrs", &attrs);
      helper.ReadAllFields(reader);
    }
  };

  // Get node entry index.
  uint32_t entry_id(uint32_t nid, uint32_t index) const {
    return node_row_ptr[nid] + index;
  }

  // Get node entry index.
  uint32_t entry_id(const Node::NodeEntry& e) const {
    return entry_id(e.node_id, e.index);
  }

  // Number of node entries
  uint32_t num_node_entries() const {
    return node_row_ptr.back();
  }

  // Number of nodes.
  uint32_t num_nodes() const {
    return static_cast<uint32_t>(nodes.size());
  }

  void Load(dmlc::JSONReader *reader) {
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("nodes", &nodes);
    helper.DeclareField("arg_nodes", &arg_nodes);
    helper.DeclareField("heads", &heads);
    helper.DeclareOptionalField("node_row_ptr", &node_row_ptr);
    helper.ReadAllFields(reader);
  }

  std::vector<Node> nodes;
  std::vector<uint32_t> arg_nodes;
  std::vector<uint32_t> node_row_ptr;
  std::vector<Node::NodeEntry> heads;
};

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_SUBGRAPH_SUBGRAPH_H_