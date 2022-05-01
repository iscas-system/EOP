import json

op_time_collection = {}

op_time_distribution = {}

def put_op_time(name, time):
    print(type(time))
    global op_time_collection
    total_op_time = 0.0
    if "total_op_time" in op_time_collection:
        total_op_time = op_time_collection.get("total_op_time")
    accumulated_time = 0.0
    if name in op_time_collection:
        accumulated_time = op_time_collection.get(name)
    accumulated_time += time[0]
    total_op_time += time[0]
    op_time_collection[name] = accumulated_time
    op_time_collection["total_op_time"] = total_op_time
    return time

def calculate_op_distribution(model_name):
    op_time_distribution["model_name"] = model_name
    for k in op_time_collection:
        if k != "total_op_time":
            op_time_distribution[k] = "%f%%" % (op_time_collection[k]/op_time_collection["total_op_time"] * 100)
    return json.dumps(op_time_distribution), json.dumps(op_time_collection)