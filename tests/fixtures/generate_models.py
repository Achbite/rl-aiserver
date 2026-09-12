"""Reproduce the two small inference fixtures with ONNX (no training).

Input: float observation[1,17]. Logits: UP=1.5-2*x, RIGHT=2*x,
all other actions=-2, where x=observation[0,0]. Value=x.
The runtime-error model instead gathers a one-element constant at int(x):
loading and x=0 succeed; x=2 produces a real ONNX Runtime index error.
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def write_model(directory, runtime_error=False):
    weights = np.zeros((17, 9), dtype=np.float32)
    weights[0, 1], weights[0, 3] = -2, 2
    bias = np.full(9, -2, dtype=np.float32)
    bias[1], bias[3] = 1.5, 0
    constants = [numpy_helper.from_array(a, n) for n, a in (
        ("weights", weights), ("bias", bias),
        ("first", np.array([0], dtype=np.int64)),
    )]
    nodes = [
        helper.make_node("MatMul", ["observation", "weights"], ["linear"]),
        helper.make_node("Add", ["linear", "bias"], ["action_logits"]),
        helper.make_node("Gather", ["observation", "first"], ["x"], axis=1),
    ]
    if runtime_error:
        constants.append(numpy_helper.from_array(np.array([0.0], dtype=np.float32), "only_value"))
        nodes += [helper.make_node("Cast", ["x"], ["index"], to=TensorProto.INT64),
                  helper.make_node("Gather", ["only_value", "index"], ["value"],
                                   name="ValueIndex", axis=0)]
    else:
        nodes.append(helper.make_node("Identity", ["x"], ["value"]))
    graph = helper.make_graph(nodes, "fixed-policy",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [1, 17])],
        [helper.make_tensor_value_info("action_logits", TensorProto.FLOAT, [1, 9]),
         helper.make_tensor_value_info("value", TensorProto.FLOAT, [1, 1])], constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, directory / ("runtime_error.onnx" if runtime_error else "fixed_policy.onnx"))


if __name__ == "__main__":
    destination = Path(__file__).resolve().parent
    write_model(destination)
    write_model(destination, runtime_error=True)
