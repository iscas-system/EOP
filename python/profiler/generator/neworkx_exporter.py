from tvm.relay.expr_functor import ExprFunctor
from tvm.relay import expr as _expr
import networkx as nx

class VisualizeExpr(ExprFunctor):
    def __init__(self):
        super().__init__()
        self.graph = nx.DiGraph()
        self.counter = 0

    def viz(self, expr):
        for param in expr.params:
            self.visit(param)

        return self.visit(expr.body)

    def visit_constant(self, const): # overload this!
        pass

    def visit_var(self, var):
        name = var.name_hint
        self.graph.add_node(name)
        self.graph.nodes[name]['style'] = 'filled'
        self.graph.nodes[name]['fillcolor'] = 'mistyrose'
        return var.name_hint

    def visit_tuple_getitem(self, get_item):
        tuple = self.visit(get_item.tuple_value)
        # self.graph.nodes[tuple]
        index = get_item.index
        # import pdb; pdb.set_trace()
        return tuple

    def visit_call(self, call):
        parents = []
        for arg in call.args:
            parents.append(self.visit(arg))
        # assert isinstance(call.op, _expr.Op)
        name = "{}({})".format(call.op.name, self.counter)
        self.counter += 1
        self.graph.add_node(name)
        self.graph.nodes[name]['style'] = 'filled'
        self.graph.nodes[name]['fillcolor'] = 'turquoise'
        self.graph.nodes[name]['shape'] = 'diamond'
        edges = []
        for i, parent in enumerate(parents):
            edges.append((parent, name, { 'label': 'arg{}'.format(i) }))
        self.graph.add_edges_from(edges)
        return name

def visualize(expr,mydir="relay_ir.png"):
    viz_expr = VisualizeExpr()
    viz_expr.viz(expr)
    graph = viz_expr.graph
    dotg = nx.nx_pydot.to_pydot(graph)
    dotg.write_png(mydir)

# {
#   "nodes": [
#     {
#       "op": "null", denotes its is a variables or input data
#       "name": "x", 
#       "inputs": []
#     }, 
#     {
#       "op": "null", 
#       "name": "w0", 
#       "inputs": []
#     }, 
#     {
#       "op": "null", 
#       "name": "w1", 
#       "inputs": []
#     }, 
#     {
#       "op": "null", 
#       "name": "w2", 
#       "inputs": []
#     }, 
#     {
#       "op": "null", 
#       "name": "w6", 
#       "inputs": []
#     }, 
#     {
#       "op": "null", 
#       "name": "w7", 
#       "inputs": []
#     }, 
#     {
#       "op": "tvm_op", denotes it is a compute op 
#       "name": "ccompiler_0", 
#       "attrs": { denotes its indexes of input and output
#         "num_outputs": "1", 
#         "num_inputs": "4", 
#         "flatten_data": "0", 
#         "func_name": "ccompiler_0"
#       }, 
#       "inputs": [
#         [ The first is the index of an op, the second may be the tuple get item index
#           0, 
#           0, 
#           0
#         ], 
#         [
#           1, 
#           0, 
#           0
#         ], 
#         [
#           2, 
#           0, 
#           0
#         ], 
#         [
#           3, 
#           0, 
#           0
#         ]
#       ]
#     }, 
#     {
#       "op": "tvm_op", 
#       "name": "tvmgen_default_fused_add_subtract_concatenate", 
#       "attrs": {
#         "num_outputs": "1", 
#         "num_inputs": "4", 
#         "flatten_data": "0", 
#         "func_name": "tvmgen_default_fused_add_subtract_concatenate", 
#         "hash": "741e7d5b70fd9dc3"
#       }, 
#       "inputs": [
#         [
#           0, 
#           0, 
#           0
#         ], 
#         [
#           4, 
#           0, 
#           0
#         ], 
#         [
#           5, 
#           0, 
#           0
#         ], 
#         [
#           6, 
#           0, 
#           0
#         ]
#       ]
#     }
#   ], 
#   "arg_nodes": [0, 1, 2, 3, 4, 5], denote it as the start point of varaibles 
#   "heads": [ unclear
#     [
#       7, 
#       0, 
#       0
#     ]
#   ], 
#   "attrs": {
#     "dltype": [data types of outputs of ops
#       "list_str", 
#       [
#         "float32", 
#         "float32", 
#         "float32", 
#         "float32", 
#         "float32", 
#         "float32", 
#         "float32", 
#         "float32"
#       ]
#     ], 
#     "device_index": [
#       "list_int", 
#       [1, 1, 1, 1, 1, 1, 1, 1]
#     ], 
#     "storage_id": [
#       "list_int", 
#       [0, 1, 2, 3, 4, 5, 6, 7]
#     ], 
#     "shape": [ shape of outputs of ops
#       "list_shape", 
#       [
#         [10, 10], 
#         [10, 10], 
#         [10, 10], 
#         [10, 10], 
#         [10, 10], 
#         [10, 10], 
#         [10, 10], 
#         [20, 10]
#       ]
#     ]
#   }, 
#   "node_row_ptr": [0, 1, 2, 3, 4, 5, 6, 7, 8] quick index of each op or variables.
# }
