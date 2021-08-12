"""
realy_graph: key class to construct RelayIR computation graph.
:author {xuyuanjia2017,huyi19}@otcaix.iscas.ac.cn
"""
import os
import sys
import tvm
import tvm.relay as relay
from tvm.relay.testing import run_infer_type
from tvm.relay.transform import gradient
import time

sys.path.append(os.path.dirname(os.getcwd()))

class op_graph:
    """
    op_graph is the key structure to maintain the computation graph.

    attributes
    ----------
    :attr dictionary: store all op's meta data
    :attr call_nodes: store computation operations
    :attr var_nodes: store variable operations
    "attr const_nodes: store constance operations
    """
    def __init__(self):
        self.dictionary = {}
        self.var_nodes = []
        self.call_nodes = []
        self.const_nodes = []
        self.tuplegetitem_nodes = []

    def insert_op(self, current_op_node):
        if self.find_if_exist(current_op_node) is None:
            self.dictionary[current_op_node.id] = current_op_node
            if current_op_node.type == "call":
                self.call_nodes.append(current_op_node)
            if current_op_node.type == "var":
                self.var_nodes.append(current_op_node)
            if current_op_node.type == "const":
                self.const_nodes.append(current_op_node)
            if current_op_node.type =="tuplegetitem":
                self.tuplegetitem_nodes.append(current_op_node)
    
    def find_starting_ops(self):
        result = []
        for temp_op in self.call_nodes:
            if_started = True 
            for key in temp_op.prior.keys():
                if temp_op.prior[key][1].type == "call" or temp_op.prior[key][1].type == "tuplegetitem":
                    if_started = False
                    break
            if if_started is True:
                result.append(temp_op)
        return result

    def find_if_exist(self, current_op_node):
        for key in self.dictionary.keys():
            if self.dictionary[key].op_instance == current_op_node.op_instance:
                return self.dictionary[key]
        return None

    def find_if_var(self, var_op):
        for temp in self.var_nodes:
            if temp.op_instance == var_op:
                return temp
        return None

    def find_if_const(self, const_op):
        for temp in self.const_nodes:
            if temp.op_instance == const_op:
                return temp
        return None
    
    def check_op_ready(self, temp_op):
        for key in temp_op.prior.keys():
            if temp_op.prior[key][1].type == "call" or temp_op.prior[key][1].type == "tuplegetitem":
                if "fw_value" not in temp_op.prior[key][1].performance_data.keys():
                    return False
                else:
                    return True
        return True

    def traverse_and_calculate_per_op(self, ir_params, x, fw=True, bw=False):
        """
        use wide-first traversial approaches to calculate each operator

        Parameters
        ----------
        :param ir_params: a tvm relayIRModule's trained parameters
        :param x: a tvm relayIRModule's input data
        :param fw: weather to profile oepration's forward process. the default is true.
        :param bw: weather to profile oepration's backward process. the default is false.
        """
        global profile_count
        available_op_queue = self.find_starting_ops()
        profile_count = 0
        while len(available_op_queue) > 0 :
            temp_op = available_op_queue.pop(0)
            if fw:
                profile_forward_relay_operator(temp_op, ir_params, x)
            if bw:
                profile_backward_relay_operator(temp_op, ir_params, x)
            for key in temp_op.next.keys():
                if self.check_op_ready(temp_op.next[key][1]):
                    available_op_queue.append(temp_op.next[key][1])
            profile_count +=1

profile_count=0
profile_point=21
op_index = 0
computation_graph = op_graph()

class op_node:
    """
    op_node is the node description class in op_graph.

    attributes
    ----------
    :attr type: may be var, const and call
    :attr id: auto-incremental and unique id
    :attr op_instance: store the reference of current op's IRModule instance.
    "attr next: the next operation dictionary of current operation
    "attr prior: the prior operation dictionary of current operation
    "attr performance_data: store the execution and memory consumpiton of current operation
    """
    def __init__(self, type, current_index, op, attrs = None):
        self.type = type
        if self.type == "var":
            self.name = op.name_hint
        if self.type == "const":
            self.name = "const"
        if self.type == "call":
            self.name = op.op.name
            self.attrs = attrs
        if self.type == "tuplegetitem":
            self.name = "tuplegetitem"
            self.tuple_value = attrs[0]
            self.index = attrs[1]
        self.id = self.name + "-" + str(current_index)
        self.op_instance = op
        self.next = {}
        self.prior = {}
        self.performance_data = {}

    def set_next(self, next_op_node, args_index):
        '''
        In the new version, the seconed arg <next_op_index> is abandoned to fix some bug!
        '''
        next_op_id = next_op_node.id
        self.next[next_op_id] = (args_index, next_op_node)
        next_op_node.prior[self.id] = (args_index, self)
    
    def print_self(self):
        #print(self.id, self.attrs, self.next, self.prior)
        print("self.id: %s" %(self.id))
        print("self.type: %s" %(self.type))
        print("self.next: %r" %(self.next))
        print("self.prior: %r" %(self.prior))


def construct_op_graph(ir_module):
    """
    use wide-first traversial approaches to construct the computation graph

    Parameters
    ----------
     :param ir_module: a tvm relayIRModule
    """
    global op_index, computation_graph
    entrance_tuple = ir_module.functions.items()[0]
    main_function = entrance_tuple[1]
    for each_param in main_function.params:
        temp_op_node = op_node("var", op_index, each_param, attrs = None)
        computation_graph.insert_op(temp_op_node)
        op_index+=1
    input = {}
    input['attrs'] = main_function.body.attrs
    input['args'] = main_function.body.args
    type = "call"
    recursive_traverse_op(type, input, temp_op=main_function.body)

def profile_memory(ir_params, x):
    computation_graph.traverse_and_calculate_per_op(ir_params, x, bw = False)


def recursive_traverse_op(attrs, args, temp_op=None):
    """
    call node means current traverser is not end, var or const node means it can end now and need to traverse new operation branches. 

    Parameters
    ----------
    :param attrs: attributes of current operation
    :param args: arguments of current operation
    :param temp_op: IRModule instance of current operation
    """
    global op_index, computation_graph
    args_index = 0
    next_op_node = op_node("call", op_index, temp_op, attrs = attrs)
    if computation_graph.find_if_exist(next_op_node) is not None :
        return computation_graph.find_if_exist(next_op_node)
    op_index+=1
    for each_arg in args:
        if isinstance(each_arg, tvm.relay.expr.Call):
            current_input = {}
            current_input['attrs'] = each_arg.attrs
            current_input['args'] = each_arg.args
            current_node_op = recursive_traverse_op("call", current_input, temp_op=each_arg)
            current_node_op.set_next(next_op_node, args_index)
            args_index += 1
        if isinstance(each_arg, tvm.relay.expr.Var):
            current_node_op = computation_graph.find_if_var(each_arg)
            if current_node_op is None:
                current_node_op = op_node("var", op_index, each_arg, attrs=None)
            current_node_op.set_next(next_op_node, args_index)
            computation_graph.insert_op(current_node_op)
            args_index += 1
        if isinstance(each_arg, tvm.relay.expr.Constant):
            current_node_op = computation_graph.find_if_const(each_arg)
            if current_node_op is None:
                current_node_op = op_node("const", op_index, each_arg, attrs=None)
            current_node_op.set_next(next_op_node, args_index)
            computation_graph.insert_op(current_node_op)
            args_index += 1
        if isinstance(each_arg, tvm.relay.expr.TupleGetItem):
            current_input = [each_arg.tuple_value, each_arg.index]
            current_node_op = recursive_traverse_op("tuplegetitem", current_input, temp_op=each_arg)
            current_node_op.set_next(next_op_node, args_index)
            args_index += 1
    computation_graph.insert_op(next_op_node)
    return next_op_node

def get_op_args(ready_op_node, dtype, params, x):
    """
    get all actual arguments of an operations. Arguments may calcuate from others. 

    Parameters
    ----------
    :param ready_op_node: an opertion that can run immendiately 
    :param dtype: argument types
    :param params: to do or optimized, not used currrently
    :param dtype: input data of a tvm realy IR
    """
    intermeidiate_args = [] # input arguments calculated from prior operations
    args_index = 0
    max_index = 0
    for key in ready_op_node.prior.keys():
        if max_index < ready_op_node.prior[key][0]:
            max_index = ready_op_node.prior[key][0]
    while args_index <= max_index:
        for key in ready_op_node.prior.keys():
            if ready_op_node.prior[key][0] == args_index:
                if ready_op_node.prior[key][1].type == "call":
                    # need to append intermeidiate args:
                    intermeidiate_args.append(ready_op_node.prior[key][1].performance_data["fw_value"])
                if ready_op_node.prior[key][1].type == "var":
                    # need to append params:
                    if ready_op_node.prior[key][1].name == '1' or ready_op_node.prior[key][1].name == 'input.1' or ready_op_node.prior[key][1].name == 'data':
                        intermeidiate_args.append(x.astype(dtype))
                    else:
                        # may be this is not necessary since its keywork arguments.
                        pass
                args_index+=1
    return intermeidiate_args

def find_nd_array_args(ready_op_node, args_index):
    """
    check weathcher an operation node's args is ready

    Parameters
    ----------
    :param ready_op_node: an opertion that can run immendiately 
    :param args_index: argument index in an operation 
    """
    for id_key in ready_op_node.prior.keys():
        temp_index = ready_op_node.prior[id_key][0]
        temp_prior_node = ready_op_node.prior[id_key][1]
        if temp_index == args_index:
            return temp_prior_node.performance_data["fw_value"].shape, temp_prior_node.performance_data["fw_value"].dtype
    print("cannot find ", ready_op_node.id, "'s intermeidiate_arg in index :", args_index)
    return None, None

def generate_intermediate_symbolic_args(ready_op_node):
    """
    get all symbolic arguments of an operations. Arguments may be replaced by var since these arguements calculated from other operations.

    Parameters
    ----------
    :param ready_op_node: an opertion that can run immendiately  
    """
    args_index = 0
    new_args = []
    for tvm_arg in ready_op_node.op_instance.args:
        if isinstance(tvm_arg, tvm.relay.expr.Call):
            s, d = find_nd_array_args(ready_op_node, args_index)
            temp_arg = tvm.relay.var(str(args_index), shape=s, dtype=d)
            new_args.append(temp_arg)
        if isinstance(tvm_arg, tvm.relay.expr.Var):
            new_args.append(tvm_arg)
        if isinstance(tvm_arg, tvm.relay.expr.Constant):
            new_args.append(tvm_arg)
        if isinstance(tvm_arg, tvm.relay.expr.TupleGetItem):
            s, d = find_nd_array_args(ready_op_node, args_index)
            temp_arg = tvm.relay.var(str(args_index), shape=s, dtype=d)
            new_args.append(temp_arg)
        args_index+=1
    return new_args

# @operation_profile
def op_forward_profile(call_interpreter, call_intput_args, ir_params):
    t0 = time.perf_counter()
    res = call_interpreter.evaluate()(*call_intput_args, **ir_params)
    t1 = time.perf_counter()
    print("running time: %s s" %(str(t1-t0)))
    return res

def profile_forward_relay_operator(ready_op_node, ir_params, x, dtype="float32"):
    """
    Sequcently compile each operaion according to its dependencies without grad.

    Parameters
    ----------
    :param ready_op_node: an opertion that can run immendiately
    :param ir_params: trained parameter from onnx or pytorch
    :param x: DL input data
    :param dtype: the default is float32

    Examples
    ----------
    new_args:[Var(0, ty=TensorType([1, 32, 224, 224], float32)), Var(8, ty=TensorType([9, 32, 3, 3], float32))]
    temp_body:  free_var %v0: Tensor[(1, 32, 224, 224), float32];
                free_var %v8: Tensor[(9, 32, 3, 3), float32];
                nn.conv2d(%v0, %v8, padding=[1, 1, 1, 1], kernel_size=[3, 3])
                def @main(%v0: Tensor[(1, 32, 224, 224), float32], %v8: Tensor[(9, 32, 3, 3), float32]) {
                    nn.conv2d(%v0, %v8, padding=[1, 1, 1, 1], kernel_size=[3, 3])
                }

    Tips:
    ----------
    the first param of tvm.relay.Function must be a list of Var.
    """
    global profile_count, profile_point
    if ready_op_node.type == "var" or ready_op_node.type == "const":
        return
    new_args = generate_intermediate_symbolic_args(ready_op_node)
    temp_body = tvm.relay.Call(ready_op_node.op_instance.op, new_args, attrs=ready_op_node.op_instance.attrs)
    call_function = tvm.relay.Function(relay.analysis.free_vars(temp_body),temp_body)
    call_functions = {"GlobalVar": None, "main": call_function}
    call_ir_module = tvm.ir.IRModule(functions=call_functions)
    with tvm.transform.PassContext(opt_level=1):
        call_interpreter = relay.build_module.create_executor("graph", call_ir_module, tvm.cuda(0), "cuda")
    call_intput_args = get_op_args(ready_op_node, dtype, ir_params, x)
    print(ready_op_node.id)

    ready_op_node.performance_data["fw_value"] = op_forward_profile(call_interpreter,call_intput_args,ir_params)
    return 

def profile_backward_relay_operator(ready_op_node, ir_params, x, dtype="float32"):
    """
    Sequcently compile each operaion according to its dependencies with auto-grad.
    
    Parameters
    ----------
    :param ready_op_node: an opertion that can run immendiately
    :param ir_params: trained parameter from onnx or pytorch
    :param x: DL input data
    :param dtype: the default is float32
    """
    global profile_count, profile_point
    if ready_op_node.type == "var" or ready_op_node.type == "const":
        return
    new_args = generate_intermediate_symbolic_args(ready_op_node)
    temp_body = tvm.relay.Call(ready_op_node.op_instance.op, new_args, attrs=ready_op_node.op_instance.attrs)
    call_function = tvm.relay.Function(relay.analysis.free_vars(temp_body), temp_body)
    call_function = run_infer_type(call_function)
    bwd_func = run_infer_type(gradient(call_function))
    call_interpreter = relay.create_executor(device = tvm.cuda(0), target = "cuda")
    call_intput_args = get_op_args(ready_op_node, dtype, ir_params, x)
    print(ready_op_node.id)
    ready_op_node.performance_data["bw_value"] = call_interpreter.evaluate(bwd_func)(*call_intput_args, **ir_params)
    return 
