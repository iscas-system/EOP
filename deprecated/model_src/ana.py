import os
import csv
'''
inceptionv3.txt
{'nn_conv2d': 72.167, 'nn_avg_pool2d': 1.021, 'nn_relu': 3.07, 'add': 8.287000000000003, 'multiply': 5.674999999999989, 'nn_dense': 0.136, 'nn_max_pool2d': 0.383, 'nn_avg_p': 0.104, 'nn_': 0.122, 'nn_softm': 0.072, 'concatenate': 0.5249999999999999, 'divide': 2.6030000000000038, 'negative': 2.445999999999995, 'sqrt': 2.518999999999996, 'nn_bias': 0.029, 'neg': 0.029, 'divi': 0.028, 'sqr': 0.027, 'nega': 0.054}
roberta.txt
{'nn_dense': 93.23200000000003, 'nn_softm': 2.163, 'cumsum': 0.102, 'nn_batch_matmul': 0.9300000000000003, 'multiply': 0.5770000000000004, 'add': 1.2389999999999974, 'erf': 0.10899999999999997, 'transpose': 0.3200000000000001, 'variance': 0.171, 'mean': 0.171, 'var': 0.036, 'mea': 0.054, 'divide': 0.10000000000000003, 'varian': 0.018, 'take': 0.028999999999999998, 'div': 0.13400000000000006, 'divi': 0.042, 'subtract': 0.14000000000000004, 'sub': 0.019000000000000003, 'tra': 0.006, 'cast': 0.023000000000000003, 'strided_s': 0.006, 'sqr': 0.11600000000000003, 'mul': 0.006, 'not': 0.006, 'repeat': 0.006, 'tan': 0.006, 'full': 0.006, 'sqrt': 0.01, 'cas': 0.005}
densenet201.txt
{'nn_conv2d': 66.72999999999999, 'add': 10.626000000000001, 'nn_': 0.783, 'multiply': 8.567999999999998, 'nn_max_p': 0.432, 'nn_softm': 0.413, 'mul': 0.223, 'nn_relu': 2.9769999999999994, 'nn_glob': 0.205, 'negative': 2.5239999999999996, 'sqrt': 2.526, 'divide': 2.6239999999999997, 'divi': 0.145, 'nn_bias_': 0.145, 'sqr': 0.142, 'neg': 0.141, 'nn_batch': 0.138}
bert-large.txt
{'nn_dense': 96.09699999999992, 'nn_softm': 1.1940000000000002, 'nn_batch_matmul': 0.5770000000000004, 'multiply': 0.29100000000000015, 'add': 0.7170000000000005, 'divide': 0.07400000000000002, 'mean': 0.11400000000000006, 'transpose': 0.14200000000000007, 'erf': 0.07200000000000002, 'variance': 0.003, 'varian': 0.14400000000000007, 'mea': 0.032999999999999995, 'take': 0.009000000000000001, 'mul': 0.09200000000000007, 'div': 0.09600000000000007, 'tra': 0.07400000000000005, 'mult': 0.004, 'sub': 0.008, 'subtract': 0.09200000000000007, 'cast': 0.002, 'strided_s': 0.002, 'sqr': 0.09200000000000007, 'tan': 0.002, 'ful': 0.002, 'rep': 0.002, 'str': 0.002, 'cas': 0.003}
transformer.txt
{'nn_dense': 69.94399999999999, 'nn_softm': 13.445, 'nn_batch': 1.135, 'nn_batch_matmul': 2.0740000000000003, 'divide': 1.0640000000000005, 'transpose': 1.0450000000000004, 'variance': 0.899, 'mean': 0.8260000000000003, 'add': 4.511000000000003, 'erf': 0.393, 'mea': 0.066, 'multiply': 1.236, 'div': 0.105, 'tak': 0.053, 'take': 0.14200000000000002, 'mult': 0.05, 'mul': 0.5379999999999999, 'sub': 0.19, 'subtract': 0.4669999999999999, 'tra': 0.23199999999999998, 'cas': 0.121, 'sqr': 0.45199999999999996, 'stri': 0.042, 'sqrt': 0.084, 'full': 0.042, 'rep': 0.041, 'str': 0.04, 'tan': 0.04}
mobilenet.txt
{'nn_conv2d': 44.88900000000002, 'add': 15.852000000000004, 'nn_relu': 6.662999999999999, 'nn_conv': 0.612, 'multiply': 15.510000000000005, 'nn_softma': 0.492, 'nn_dense': 0.49, 'nn_global': 0.279, 'divide': 4.536, 'nn_bia': 0.175, 'sqrt': 4.470000000000001, 'negative': 4.266999999999999, 'negati': 0.17, 'divi': 0.169, 'nn_batch': 0.165, 'mul': 0.162}
resnet152.txt
{'nn_conv2d': 61.601000000000006, 'nn_dense': 0.117, 'add': 12.826000000000016, 'multiply': 10.525999999999918, 'nn_': 0.165, 'nn_relu': 4.045999999999998, 'nn_max_p': 0.06, 'divide': 3.2829999999999937, 'negative': 3.152999999999999, 'sqrt': 3.2179999999999946, 'div': 0.023, 'negati': 0.022, 'nn_bias_': 0.022, 'nn_batch': 0.02}
vgg16.txt
{'nn_dense': 15.878, 'nn_conv2d': 71.966, 'nn_conv2': 1.169, 'nn_relu': 4.645, 'nn_bias_': 1.91, 'nn_max_p': 0.599, 'nn_bias_add': 2.789, 'nn_max_pool2d': 0.746, 'nn_softmax': 0.211, 'nn_batc': 0.089}
resnet101.txt
{'nn_conv2d': 63.24600000000001, 'nn_dense': 0.158, 'add': 12.295999999999982, 'nn_': 0.268, 'multiply': 10.07300000000002, 'nn_relu': 3.867999999999999, 'nn_max_p': 0.083, 'divide': 3.1499999999999915, 'sqrt': 3.0339999999999963, 'negative': 2.9689999999999954, 'nn_batch': 0.03, 'nega': 0.03}
'''

dir_path = "/root/github/TVMProfiler/deprecated/model_src/datat"
data_list = os.listdir(dir_path)
for i in range(len(data_list)):
    file_path = os.path.join(dir_path,data_list[i])
    if os.path.isfile(file_path):
        with open(file_path,'r') as f:
            eachlines = f.readlines()
            # op = []
            # op_proportion = []
            op = {}
            for j in range(len(eachlines)):
                line = eachlines[j]
                items = []
                item = ''
                for k in range(len(line)):

                    if line[k] != ' ':
                        item += line[k]
                    else:
                        if len(item)>0:
                            items.append(item)
                        if len(items) == 4:
                            break
                        item =''
                proportion = float(items[3])
                op_fullname = items[1][21:]
                if op_fullname[-1] >= '0' and op_fullname[-1] <= '9':
                    tmp = len(op_fullname)
                    while op_fullname[tmp-1] != '_':
                        tmp -= 1
                    op_fullname = op_fullname[0:tmp-1]
                if op_fullname in op.keys():
                    op[op_fullname] += proportion
                else:
                    op[op_fullname] = proportion

            raw_data = []
            cnt = 0
            for key in op.keys():
                if op[key]<1:
                    continue
                raw_data.append({})
                raw_data[cnt]['op_name'] = key
                raw_data[cnt]['running time percentage'] = op[key]
                cnt += 1
            out_file = os.path.join("/root/github/TVMProfiler/deprecated/model_src/datac",data_list[i].split('.')[0]+'.csv')
            with open(out_file, 'w', newline='', encoding='utf-8') as f:
                header = ["op_name","running time percentage"]
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(raw_data)



