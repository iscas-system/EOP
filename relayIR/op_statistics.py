op_time_collection = {}

op_time_distribution = {}

def put_op_time(name, time):
    global op_time_collection
    total_op_time = 0.0
    if op_time_collection.has_key("total_op_time"):
        total_op_time = op_time_collection.get("total_op_time")
    accumulated_time = 0.0
    if op_time_collection.has_key(name):
        accumulated_time = op_time_collection.get(name)
    accumulated_time += time
    total_op_time += time
    op_time_collection[name] = accumulated_time
    op_time_collection["total_op_time"] = total_op_time

def calculate_op_distribution(model_name):
    op_time_distribution["model_name"] = model_name
    for k, v in op_time_collection:
        if k != "total_op_time":
            op_time_distribution[k] = "%.2f%%" % (op_time_collection[k]/op_time_collection["total_op_time"] * 100)