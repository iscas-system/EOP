"""
realy_graph: key class to construct RelayIR computation graph.
:author {xuyuanjia2017,huyi19}@otcaix.iscas.ac.cn
"""
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
# from std_memory_profiler import profile, operation_time_profile, operation_memory_profile, operation_cuda_memory_profile
from tvm.relay.testing import check_grad, run_infer_type
from tvm.relay.transform import gradient
import time
import sys
import csv
from cnn_workload_generator import get_network, compile_without_log, create_graph_executor_on_single_device, evaluate_time_with_tvm_evaluator, create_operator_executor_on_single_device
from op_statistics import put_op_time
import json

key_op_list = {}

def op_store(tmp_op,tmp_args,tmp_params,tmp_list):
    global key_op_list
    if tmp_op.type != "call":
        return
    key_op_name = tmp_op.id.split('-')[0]
    if key_op_name not in key_op_list.keys():
        key_op_list[key_op_name] = {}
        key_op_list[key_op_name]['args'] = []
        key_op_list[key_op_name]['params'] = []
    args_shape = []
    for i in range(len(tmp_args)):
        args_shape.append(list(tmp_args[i].shape))
    key_op_list[key_op_name]['args'].append(args_shape)
    params_shape = []
    for i in range(len(tmp_args),len(tmp_list)):
        params_shape.append(list(tmp_params[tmp_list[i].name_hint].shape))
    key_op_list[key_op_name]['params'].append(params_shape)

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
        self.tuple_nodes = []

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
            if current_op_node.type =="tuple":
                self.tuple_nodes.append(current_op_node)

    def find_starting_ops(self):
        result = []
        for temp_op in self.call_nodes:
            if_started = True
            for key in temp_op.prior.keys():
                if temp_op.prior[key][1].type == "call" or temp_op.prior[key][1].type == "tuplegetitem" or temp_op.prior[key][1].type == "tuple":
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
            if temp_op.prior[key][1].type == "call" or temp_op.prior[key][1].type == "tuplegetitem" or temp_op.prior[key][1].type == "tuple":
                #to do
                if "fw_value" not in temp_op.prior[key][1].performance_data.keys():
                    return False
        return True

    def traverse_and_calculate_per_op(self, ir_params, x, input_name, device, target, fw=True, bw=False, output_file = "out.csv"):
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
        print(len(available_op_queue))
        # for p in available_op_queue:
        #     print(p.id)
        #     for key in p.prior.keys():
        #         print("prev: %s" %(p.prior[key][1].id))
        profile_count = 0
        output_list = []
        while len(available_op_queue) > 0:
            temp_op = available_op_queue.pop(0)
            if "fw_value"  in temp_op.performance_data.keys():
                continue
            print("temp_op: %s" %(temp_op.id))
            op_list = [temp_op]
            output = {}
            temp_op_list = []
            if temp_op.concentrate != False:
                for key1 in temp_op.next.keys():
                    op_list.append(temp_op.next[key1][1])
                    tmp = key1
                    temp_op_list.append(temp_op.next[key1][1])
                temp_op = temp_op.next[tmp][1]
            profile_forward_relay_operator_time(op_list, ir_params, x, input_name, device, target)
            #print("nextlen %r" %(len(temp_op.next)))
            if len(temp_op_list) > 0:
                for p in temp_op_list:
                    for key in p.next.keys():
                        #print("%r:%r" % (temp_op.next[key][1].id, self.check_op_ready(temp_op.next[key][1])))
                        if self.check_op_ready(p.next[key][1]) and  "fw_value" not in p.next[key][1].performance_data.keys():
                            available_op_queue.append(p.next[key][1])
            else:
                for key in temp_op.next.keys():
                    #print("%r:%r" % (temp_op.next[key][1].id, self.check_op_ready(temp_op.next[key][1])))
                    if self.check_op_ready(temp_op.next[key][1]) and "fw_value" not in temp_op.next[key][1].performance_data.keys():
                        available_op_queue.append(temp_op.next[key][1])
            profile_count +=1
            if profile_count == 1:
                print(key_op_list)
        # with open(output_file,"w") as f:
        #     json.dump(key_op_list,f)
        return json.dumps(key_op_list)


profile_count=0
profile_point=21
op_index = 0
computation_graph = op_graph()

tmpcnt = 0

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
            self.data = attrs
        if self.type == "call":
            self.name = op.op.name
            self.attrs = attrs
        if self.type == "tuplegetitem":
            #to do
            self.name = "tuplegetitem"
            self.tuple_value = attrs[0]
            self.index = attrs[1]
        if self.type == "tuple":
            self.name = "tuple"
            self.fields = op.fields
        self.id = self.name + "-" + str(current_index)
        self.op_instance = op
        self.next = {}
        self.prior = {}
        self.performance_data = {}
        self.concentrate = False
        #self.tuplegetitem_cnt = 0
        self.fwmetadata = {}
        self.bwmetadata = {}
        self.body = None

    def set_next(self, next_op_node, args_index):
        '''
        In the new version, the seconed arg <next_op_index> is abandoned to fix some bug!
        '''
        next_op_id = next_op_node.id
        self.next[next_op_id] = (args_index, next_op_node)
        next_op_node.prior[self.id] = (args_index, self)
        if isinstance(next_op_node.op_instance, tvm.relay.expr.TupleGetItem):
            #self.tuplegetitem_cnt += 1
            self.concentrate = True

    def print_self(self):
        print(self.id, self.attrs, self.next, self.prior)
        print("self.id: %s" %(self.id))
        print("self.type: %s" %(self.type))
        print("self.next: %r" %(self.next))
        print("self.prior: %r" %(self.prior))
        print("self.attrs: %r" %(self.attrs))
        print("self.attrs:")
        '''
        global tmpcnt
        if tmpcnt == 0:
            help(self.attrs)
            tmpcnt += 1
        '''
        for tmp in self.attrs.keys():
            print("key: %r" %(tmp))
            print("value: %r" %(self.attrs.get_str(tmp)))
        return

def start_iteration(main_function):
    def case1():
        input = {}
        input['attrs'] = main_function.body.attrs
        input['args'] = main_function.body.args
        type = "call"
        recursive_traverse_op(type, input, temp_op=main_function.body)
    def case2():
        input = [main_function.body.tuple_value, main_function.body.index]
        type = "tuplegetitem"
        recursive_traverse_op(type, input, temp_op=main_function.body)
    def case3():
        input = main_function.body.fields
        type = "tuple"
        recursive_traverse_op(type, input, temp_op=main_function.body)
    def default():
        print("unknown main_function!")
    switch = {tvm.relay.expr.Call:case1,
              tvm.relay.expr.TupleGetItem:case2,
              tvm.relay.expr.Tuple:case3}
    switch.get(type(main_function.body), default)()

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
    start_iteration(main_function)
    # for key in computation_graph.dictionary.keys():
    #     print(computation_graph.dictionary[key].id)
        # if key == "tuplegetitem-40":
        #     current_op_node = computation_graph.dictionary[key]
        #     break
    #if current_op_node.id == "tuple-38":
    # print('son')
    # for key in current_op_node.next.keys():
    #     print(current_op_node.next[key][1].id)
    # print('father')
    # for key in current_op_node.prior.keys():
    #     print(current_op_node.prior[key][1].id)

def profile_resource_usage(ir_params, x, input_name, device=tvm.cuda(0), target="cuda", output_file = "out.csv"):
    return computation_graph.traverse_and_calculate_per_op(ir_params, x, input_name, device, target, bw = False, output_file = output_file)


def recursive_traverse_op(type, input, temp_op=None):
    """
    call node means current traverser is not end, var or const node means it can end now and need to traverse new operation branches.

    Parameters
    ----------
    :param attrs: attributes of current operation
    :param args: arguments of current operation
    :param temp_op: IRModule instance of current operation
    """
    global op_index, computation_graph
    args = []
    args_index = 0
    if type == "call":
        next_op_node = op_node("call", op_index, temp_op, attrs = input['attrs'])
        args = input['args']
    if type == "tuplegetitem":
        #to do
        next_op_node = op_node("tuplegetitem", op_index, temp_op, attrs = input)
        args = input
    if type == "tuple":
        next_op_node = op_node("tuple", op_index, temp_op, attrs=input)
        args = input
    if computation_graph.find_if_exist(next_op_node) != None :
        return computation_graph.find_if_exist(next_op_node)
    op_index+=1

    for each_arg in args:
        if isinstance(each_arg, tvm.relay.expr.Call):
            current_input = {}
            current_input['attrs'] = each_arg.attrs
            current_input['args'] = each_arg.args
            current_node_op = recursive_traverse_op("call", current_input, temp_op = each_arg)
            current_node_op.set_next(next_op_node, args_index)
            args_index+=1
        if isinstance(each_arg,tvm.relay.expr.Var):
            current_node_op = computation_graph.find_if_var(each_arg)
            if current_node_op == None:
                current_node_op = op_node("var", op_index, each_arg, attrs = None)
            current_node_op.set_next(next_op_node, args_index)
            computation_graph.insert_op(current_node_op)
            args_index+=1
        if isinstance(each_arg,tvm.relay.expr.Constant):
            current_node_op = computation_graph.find_if_const(each_arg)
            if current_node_op == None:
                current_node_op = op_node("const", op_index, each_arg, attrs=each_arg.data)
            current_node_op.set_next(next_op_node, args_index)
            computation_graph.insert_op(current_node_op)
            args_index += 1
        if isinstance(each_arg,tvm.relay.expr.TupleGetItem):
            current_input = [each_arg.tuple_value,each_arg.index]
            current_node_op = recursive_traverse_op("tuplegetitem", current_input, temp_op = each_arg)
            current_node_op.set_next(next_op_node, args_index)
            args_index += 1
        if isinstance(each_arg, tvm.relay.expr.Tuple):
            current_input = each_arg.fields
            current_node_op = recursive_traverse_op("tuple", current_input, temp_op = each_arg)
            current_node_op.set_next(next_op_node, args_index)
            args_index+=1

    computation_graph.insert_op(next_op_node)

    return next_op_node

def view_computationgraph():
    global computation_graph


def generate_intermediate_actual_args(ready_op_node, dtype, x, input_name):
    """
    get all actual arguments of an operations. Arguments may calcuate from others.

    Parameters
    ----------
    :param ready_op_node: an opertion that can run immendiately
    :param dtype: argument types
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
                if ready_op_node.prior[key][1].type == "tuple":
                    if isinstance(ready_op_node.prior[key][1].performance_data["fw_value"], list):
                        for temp in ready_op_node.prior[key][1].performance_data["fw_value"]:
                            intermeidiate_args.append(temp)
                if ready_op_node.prior[key][1].type == "call" or ready_op_node.prior[key][1].type == "tuplegetitem":
                    # need to append intermeidiate args:
                    intermeidiate_args.append(ready_op_node.prior[key][1].performance_data["fw_value"])
                if ready_op_node.prior[key][1].type == "var":
                    # need to append params:
                    #if ready_op_node.prior[key][1].name == '1' or ready_op_node.prior[key][1].name == 'input.1' or ready_op_node.prior[key][1].name == 'data' \
                    #or ready_op_node.prior[key][1].name in input_name:
                    if ready_op_node.prior[key][1].name in input_name:
                        tmp = x[ready_op_node.prior[key][1].name]
                        intermeidiate_args.append(tmp.astype(dtype))
                    else:
                        # may be this is not necessary since its keywork arguments.
                        pass
                # if ready_op_node.prior[key][1].type == "const" and :
                #     intermeidiate_args.append(ready_op_node.prior[key][1].data)
                break
        args_index+=1

    return intermeidiate_args

def find_call_value(ready_op_node, args_index):
    """
    find an operation' input calculated from other operation (relay call). Other kinds of input shold be considered from other methods.

    Parameters
    ----------
    :param ready_op_node: an opertion that can run immendiately
    :param args_index: argument index in an operation
    """
    for id_key in ready_op_node.prior.keys():
        temp_index = ready_op_node.prior[id_key][0]
        temp_prior_node = ready_op_node.prior[id_key][1]
        #print(temp_prior_node.id)
        #print("%r/%r" %(temp_prior_node.performance_data["fw_value"].shape, temp_prior_node.performance_data["fw_value"].dtype))
        if temp_index == args_index:
            if type(temp_prior_node.performance_data["fw_value"]) is list:
                ret = []
                for m in temp_prior_node.performance_data["fw_value"]:
                    ret.append((m.shape,m.dtype))
                return ret
            else:
                return [(temp_prior_node.performance_data["fw_value"].shape, temp_prior_node.performance_data["fw_value"].dtype)]
    print("cannot find ", ready_op_node.id, "'s intermeidiate_arg in index :", args_index)
    return None

def find_tuple_value(ready_op_node, args_index):
    for id_key in ready_op_node.prior.keys():
        temp_index = ready_op_node.prior[id_key][0]
        temp_prior_node = ready_op_node.prior[id_key][1]
        if temp_index == args_index:
            ret = []
            for m in temp_prior_node.performance_data["fw_value"]:
                ret.append((m.shape,m.dtype))
            return ret
    print("cannot find ", ready_op_node.id, "'s intermeidiate_arg in index :", args_index)
    return None

def generate_intermediate_symbolic_args(ready_op_node):
    """
    get all symbolic arguments of an operations. Arguments may be replaced by var since these arguements calculated from other operations.

    Parameters
    ----------
    :param ready_op_node: an opertion that can run immendiately
    """
    args_index = 0
    new_args = []
    range_args = None
    if isinstance(ready_op_node.op_instance, tvm.relay.expr.Tuple):
        range_args = ready_op_node.op_instance.fields
    else :
        range_args = ready_op_node.op_instance.args
    for tvm_arg in range_args:
        #print("tvm_arg: %r" %(tvm_arg))
        if isinstance(tvm_arg, tvm.relay.expr.Call):
            result = find_call_value(ready_op_node, args_index)
            if len(result) == 1:
                m = result[0]
                temp_arg = tvm.relay.var(str(args_index), shape=m[0], dtype=m[1])
            if len(result) > 1:
                temp_arg = []
                for m in result:
                    temp_arg.append(tvm.relay.var(str(args_index), shape=m[0], dtype=m[1]))
                    args_index += 1
                args_index -= 1
                temp_arg = tvm.relay.expr.Tuple(temp_arg)
            new_args.append(temp_arg)
        if isinstance(tvm_arg, tvm.relay.expr.Var):
            new_args.append(tvm_arg)
        if isinstance(tvm_arg, tvm.relay.expr.Constant):
            new_args.append(tvm_arg)
        if isinstance(tvm_arg, tvm.relay.expr.TupleGetItem):
            result = find_call_value(ready_op_node, args_index)
            if len(result) == 1:
                m = result[0]
                temp_arg = tvm.relay.var(str(args_index), shape=m[0], dtype=m[1])
            if len(result) > 1:
                temp_arg = []
                for m in result:
                    temp_arg.append(tvm.relay.var(str(args_index), shape=m[0], dtype=m[1]))
                    args_index += 1
                args_index -= 1
                temp_arg = tvm.relay.expr.Tuple(temp_arg)
            new_args.append(temp_arg)
        if isinstance(tvm_arg, tvm.relay.expr.Tuple):
            result = find_tuple_value(ready_op_node,args_index)
            temp_arg = []
            for m in result:
                temp_arg.append(tvm.relay.var(str(args_index), shape=m[0], dtype=m[1]))
                args_index += 1
            args_index -= 1
            temp_arg = tvm.relay.expr.Tuple(temp_arg)
            new_args.append(temp_arg)
        args_index+=1
    #print(len(new_args))
    return new_args

def generator_operation_profile_meta(temp_body):
    temp_dict = {}
    temp_body2 = temp_body
    if isinstance(temp_body, tvm.relay.expr.TupleGetItem):
        temp_body2 = temp_body.tuple_value
    temp_dict["name"] = temp_body2.op
    temp_dict["args"] = temp_body2.args
    temp_dict["attrs"] = temp_body2.attrs
    return temp_dict

def profile_forward_relay_operator(ready_op_node_list, ir_params, x, input_name, device, target, dtype="float32"):
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
    op_list_len = len(ready_op_node_list)
    ready_op_node = ready_op_node_list[0]
    if op_list_len == 1:
        if ready_op_node.type == "var" or ready_op_node.type == "const":
            #to do
            return
    new_args = generate_intermediate_symbolic_args(ready_op_node)
    temp_body = None
    if isinstance(ready_op_node.op_instance, tvm.relay.expr.Tuple):
        temp_body = tvm.relay.expr.Tuple(new_args)
    else :
        temp_body = tvm.relay.Call(ready_op_node.op_instance.op, new_args, attrs=ready_op_node.op_instance.attrs, type_args=ready_op_node.op_instance.type_args)
    body_list = []
    for i in range(1,op_list_len):
        if i>1:
            for tkey in ready_op_node_list[i].prior.keys():
                if ready_op_node_list[i].prior[tkey][1].id == ready_op_node.id:
                    if ready_op_node_list[i].type == "tuplegetitem":
                        body_list.append(tvm.relay.expr.TupleGetItem(temp_body,ready_op_node_list[i].op_instance.index))
                    if ready_op_node_list[1].type == "call":
                        body_list.append(tvm.relay.expr.Call(ready_op_node[i].op_instance.op,
                                                             generate_intermediate_symbolic_args(ready_op_node)[i],
                                                             attrs=ready_op_node[i].op_instance.attrs,
                                                             type_args=ready_op_node[i].op_instance.type_args))
                else:
                    body_list[0] = tvm.relay.expr.TupleGetItem(body_list[0],ready_op_node_list[i].op_instance.index)
        else:
            body_list.append(tvm.relay.expr.TupleGetItem(temp_body,ready_op_node_list[i].op_instance.index))
        #temp_body = tvm.relay.expr.TupleGetItem(temp_body,ready_op_node_list[i].op_instance.index)
    #print("attrs: %r" %(ready_op_node.op_instance.attrs))
    #print(temp_body)
    if len(body_list) == 1:
        temp_body = body_list[0]
    if len(body_list) > 1:
        temp_body = tvm.relay.expr.Tuple(body_list)

    #print("hello world")
    call_function = tvm.relay.Function(relay.analysis.free_vars(temp_body),temp_body)
    call_functions = {"main": call_function}
    call_ir_module = tvm.ir.IRModule(functions=call_functions)
    with tvm.transform.PassContext(opt_level=1):
        call_interpreter = relay.build_module.create_executor("graph", call_ir_module, device, target)
    call_intput_args = generate_intermediate_actual_args(ready_op_node, dtype, x, input_name)
    # ready_op_node.print_self()
    # print(dir(ready_op_node.op_instance))
    # print("first: ")
    # print(*call_intput_args)
    # print("second: ")
    # print(**ir_params)
    # for key in ready_op_node.prior.keys():
    #     print(ready_op_node.prior[key][1].name)
    # print("op_params:")
    # tmp_param = call_interpreter.mod["main"].params
    # cnt = 0
    # out = ""
    # for p in tmp_param:
    #     out = out + str(p.name_hint) + ','
    #     cnt = cnt + 1
    #     if cnt % 10 == 0:
    #         print(out)
    #         out = ""
    # if cnt % 10 != 0:
    #     print(out)
    # print("args:")
    # for p in call_intput_args:
    #     print(p)
    # print("params:")
    # for p in ir_params:
    #     print(p)
    #'''
    # print("break %r" %(ready_op_node.id))

    metadata = {}

    if len(call_intput_args) > 0 :
        metadata['fw_inputsize'] = 0
        for p in call_intput_args:
            metadata['fw_inputsize'] += sys.getsizeof(p)
    else:
        metadata['fw_inputsize'] = 0

    if target == "llvm":
        @operation_memory_profile(stream=sys.stdout, operation_meta=metadata)
        def op_memory_forward_profile(call_interpreter, call_intput_args, ir_params):
            return call_interpreter.evaluate()(*call_intput_args, **ir_params)

        op_memory_forward_profile(call_interpreter, call_intput_args, ir_params)

    if target == "cuda":
        @operation_cuda_memory_profile(stream=sys.stdout, operation_meta=metadata)
        def op_memory_forward_profile(call_interpreter, call_intput_args, ir_params):
            return call_interpreter.evaluate()(*call_intput_args, **ir_params)

        op_memory_forward_profile(call_interpreter, call_intput_args, ir_params)

    @operation_time_profile(stream=sys.stdout, operation_meta=metadata)
    def op_time_forward_profile(call_interpreter, call_intput_args, ir_params):
        return call_interpreter.evaluate()(*call_intput_args, **ir_params)

    ready_op_node.performance_data["fw_value"] = op_time_forward_profile(call_interpreter,call_intput_args,ir_params)
    for i in range(1,op_list_len):
        if type(ready_op_node.performance_data["fw_value"]) is list:
            if isinstance(ready_op_node_list[i].op_instance,tvm.relay.expr.TupleGetItem):
                index = ready_op_node_list[i].op_instance.index
                ready_op_node_list[i].performance_data["fw_value"] = ready_op_node.performance_data["fw_value"][index]
            else:
                ready_op_node_list[i].performance_data["fw_value"] = ready_op_node.performance_data["fw_value"]
        else:
            ready_op_node_list[i].performance_data["fw_value"] = ready_op_node.performance_data["fw_value"]

    for i in range(op_list_len):
        for key in metadata.keys():
            ready_op_node_list[i].fwmetadata[key] = metadata[key]

    print(ready_op_node.id)
    #print(ready_op_node.fwmetadata)
    return ready_op_node.id,metadata


def profile_forward_relay_operator_time(ready_op_node_list, ir_params, x, input_name, device, target, dtype="float32"):
    global profile_count, profile_point
    op_list_len = len(ready_op_node_list)
    ready_op_node = ready_op_node_list[0]
    print("current_op:"+ready_op_node.id)
    # print(type(ready_op_node.op_instance))
    print("father:")
    for temp in ready_op_node.prior.keys():
        print(ready_op_node.prior[temp][1].id)
    print("son:")
    for temp in ready_op_node.next.keys():
        print(ready_op_node.next[temp][1].id)

    if op_list_len == 1:
        if ready_op_node.type == "var" or ready_op_node.type == "const":
            #to do
            return
    new_args = generate_intermediate_symbolic_args(ready_op_node)
    # print("symbolic args:")
    # for temp in new_args:
    #     print(type(temp))
    temp_body = None
    if isinstance(ready_op_node.op_instance, tvm.relay.expr.Tuple):
        temp_body = tvm.relay.expr.Tuple(new_args)
    else :
        temp_body = tvm.relay.Call(ready_op_node.op_instance.op, new_args, attrs=ready_op_node.op_instance.attrs, type_args=ready_op_node.op_instance.type_args)

    body_list = []
    for i in range(1,op_list_len):
        if i>1:
            for tkey in ready_op_node_list[i].prior.keys():
                if ready_op_node_list[i].prior[tkey][1].id == ready_op_node.id:
                    body_list.append(tvm.relay.expr.TupleGetItem(temp_body,ready_op_node_list[i].op_instance.index))
                else:
                    body_list[0] = tvm.relay.expr.TupleGetItem(body_list[0],ready_op_node_list[i].op_instance.index)
        else:
            body_list.append(tvm.relay.expr.TupleGetItem(temp_body,ready_op_node_list[i].op_instance.index))
    if len(body_list) == 1:
        temp_body = body_list[0]
    if len(body_list) > 1:
        temp_body = tvm.relay.expr.Tuple(body_list)

    call_function = tvm.relay.Function(relay.analysis.free_vars(temp_body),temp_body)
    call_functions = {"main": call_function}
    call_ir_module = tvm.ir.IRModule(functions=call_functions)
    print(call_ir_module)
    with tvm.transform.PassContext(opt_level=1):
        call_interpreter = relay.build_module.create_executor("graph", call_ir_module, device, target)
    call_intput_args = generate_intermediate_actual_args(ready_op_node, dtype, x, input_name)

    # print("actual input args:")
    # for temp in call_intput_args:
    #     print(type(temp))
    # print(call_ir_module)

    # for key in ready_op_node.prior.keys():
    #     print(ready_op_node.prior[key][1].name)
    # print("op_params:")
    # tmp_param = call_interpreter.mod["main"].params
    # cnt = 0
    # out = ""
    # for p in tmp_param:
    #     out = out + str(p.name_hint) + ','
    #     cnt = cnt + 1
    #     if cnt % 10 == 0:
    #         print(out)
    #         out = ""
    # if cnt % 10 != 0:
    #     print(out)
    # print("args:")
    # for p in call_intput_args:
    #     print(type(p))
    # print("params:")
    # for p in ir_params:
    #     print(p)

    tmp_param = call_interpreter.mod["main"].params
    if len(tmp_param)>1:
        for i in range(len(tmp_param)-1):
            if tmp_param[i].name_hint in ir_params.keys() and tmp_param[i+1].name_hint not in ir_params.keys():
                new_call_intput_args = []
                cnt = 0
                for beg in range(len(tmp_param)):
                    if tmp_param[beg].name_hint in ir_params.keys():
                        new_call_intput_args.append(ir_params[tmp_param[beg].name_hint])
                    else:
                        if cnt < len(call_intput_args):
                            new_call_intput_args.append(call_intput_args[cnt])
                            cnt += 1
                        else:
                            raise Exception("too many arguments!")
                # new_call_intput_args = []
                # for s in range(len(new_call_intput_args1)):
                #     new_call_intput_args.append(new_call_intput_args1[s])
                # for s in range(len(new_call_intput_args2)):
                #     new_call_intput_args.append(new_call_intput_args2[s])
                ir_params = {}
                call_intput_args = new_call_intput_args
                break
    op_store(ready_op_node, call_intput_args, ir_params, tmp_param)
    ready_op_node.performance_data["fw_value"] = call_interpreter.evaluate()(*call_intput_args, **ir_params)
    # ready_op_node.performance_data["fw_value"] = call_interpreter.evaluate()(*call_intput_args, **ir_params)

    # metadata = {}
    #
    # if len(call_intput_args) > 0:
    #     metadata['fw_inputsize'] = 0
    #     for p in call_intput_args:
    #         metadata['fw_inputsize'] += sys.getsizeof(p)
    #     tmp_param = call_interpreter.mod["main"].params
    #     for p in tmp_param:
    #         if p in ir_params.keys():
    #             metadata['fw_inputsize'] += sys.getsizeof(ir_params[p])
    # else:
    #     metadata['fw_inputsize'] = 0
    #
    # if target == "llvm":
    #     @operation_memory_profile(stream=sys.stdout, operation_meta=metadata)
    #     def op_memory_forward_profile(call_interpreter, call_intput_args, ir_params):
    #         return call_interpreter.evaluate()(*call_intput_args, **ir_params)
    #
    #     op_memory_forward_profile(call_interpreter, call_intput_args, ir_params)
    #
    # if target == "cuda":
    #     @operation_cuda_memory_profile(stream=sys.stdout, operation_meta=metadata)
    #     def op_memory_forward_profile(call_interpreter, call_intput_args, ir_params):
    #         return call_interpreter.evaluate()(*call_intput_args, **ir_params)
    #
    #     op_memory_forward_profile(call_interpreter, call_intput_args, ir_params)
    #
    # ready_op_node.performance_data["fw_value"] = call_interpreter.evaluate()(*call_intput_args, **ir_params)
    # print("return value:")
    # print(type(ready_op_node.performance_data["fw_value"]))
    for i in range(1, op_list_len):
        if type(ready_op_node.performance_data["fw_value"]) is list:
            if isinstance(ready_op_node_list[i].op_instance, tvm.relay.expr.TupleGetItem):
                index = ready_op_node_list[i].op_instance.index
                ready_op_node_list[i].performance_data["fw_value"] = ready_op_node.performance_data["fw_value"][index]
            else:
                ready_op_node_list[i].performance_data["fw_value"] = ready_op_node.performance_data["fw_value"]
        else:
            ready_op_node_list[i].performance_data["fw_value"] = ready_op_node.performance_data["fw_value"]
    lib = compile_without_log(call_ir_module, target, ir_params)
    actual_module = create_operator_executor_on_single_device(lib, call_intput_args, target)
    # result = put_op_time(ready_op_node.name, evaluate_time_with_tvm_evaluator(actual_module, device))

    # print(ready_op_node.id)

def profile_backward_relay_operator(ready_op_node_list, ir_params, x, input_name, device, target, dtype="float32"):
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
    op_list_len = len(ready_op_node_list)
    ready_op_node = ready_op_node_list[0]
    if op_list_len == 1:
        if ready_op_node.type == "var" or ready_op_node.type == "const":
            # to do
            return
    new_args = generate_intermediate_symbolic_args(ready_op_node)
    temp_body = tvm.relay.Call(ready_op_node.op_instance.op, new_args, attrs=ready_op_node.op_instance.attrs,
                               type_args=ready_op_node.op_instance.type_args)
    for i in range(1, op_list_len):
        temp_body = tvm.relay.expr.TupleGetItem(temp_body, ready_op_node_list[i].op_instance.index)
    # print("attrs: %r" %(ready_op_node.op_instance.attrs))
    call_function = tvm.relay.Function(relay.analysis.free_vars(temp_body), temp_body)
    call_function = run_infer_type(call_function)
    bwd_func = run_infer_type(gradient(call_function))
    call_interpreter = relay.create_executor(device = device, target = target)
    call_intput_args = generate_intermediate_actual_args(ready_op_node, dtype, x, input_name)

    metadata = {}
    if len(call_intput_args) > 0 :
        metadata['bw_inputsize'] = 0
        for p in call_intput_args:
            metadata['bw_inputsize'] += sys.getsizeof(p)
    else:
        metadata['bw_inputsize'] = 0

    if target == "llvm":
        @operation_memory_profile(stream=sys.stdout, operation_meta=metadata)
        def op_memory_backward_profile(call_interpreter, call_intput_args, ir_params, bwd_func_):
            return call_interpreter.evaluate(bwd_func_)(*call_intput_args, **ir_params)

        op_memory_backward_profile(call_interpreter, call_intput_args, ir_params, bwd_func)

    if target == "cuda":
        @operation_cuda_memory_profile(stream=sys.stdout, operation_meta=metadata)
        def op_memory_backward_profile(call_interpreter, call_intput_args, ir_params, bwd_func_):
            return call_interpreter.evaluate(bwd_func_)(*call_intput_args, **ir_params)

        op_memory_backward_profile(call_interpreter, call_intput_args, ir_params, bwd_func)

    @operation_time_profile(stream=sys.stdout, operation_meta=metadata)
    def op_time_backward_profile(call_interpreter, call_intput_args, ir_params, bwd_func_):
        return call_interpreter.evaluate(bwd_func_)(*call_intput_args, **ir_params)
    ready_op_node.performance_data["bw_value"] = op_time_backward_profile(call_interpreter,call_intput_args,ir_params,bwd_func)
    for i in range(1, op_list_len):
        if type(ready_op_node.performance_data["fw_value"]) is list:
            if isinstance(ready_op_node_list[i].op_instance, tvm.relay.expr.TupleGetItem):
                index = ready_op_node_list[i].op_instance.index
                ready_op_node_list[i].performance_data["fw_value"] = ready_op_node.performance_data["fw_value"][index]
            else:
                ready_op_node_list[i].performance_data["fw_value"] = ready_op_node.performance_data["fw_value"]
        else:
            ready_op_node_list[i].performance_data["fw_value"] = ready_op_node.performance_data["fw_value"]

    for i in range(op_list_len):
        for key in metadata.keys():
            ready_op_node_list[i].bwmetadata[key] = metadata[key]

    print(ready_op_node.id)
    #print(ready_op_node.bwmetadata)

    return ready_op_node.id,metadata
