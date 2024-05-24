import pandas as pd
#csv转换，将标签多数类标签设为0并放在最前面，其他标签设为1，2，等并依次向后
from collections import Counter
def replace_elements(lst):
    counter = dict(Counter(lst).most_common())
    mapping = {key: i for i, key in enumerate(counter.keys())}
    return [mapping[x] for x in lst]
# 读取CSV文件
def transfor_data(data):
    label=data.columns[-1]
    data = data.sort_values(by=label)
    l1=[]
    for i in data[label]:
        i=float(i)
        l1.append(i)
    num_classes = len(set(l1))
    l_1=replace_elements(l1)
    data.iloc[:, -1] = l_1
    data = data.sort_values(by=label)
    data= data.reset_index(drop=True)
    return data,num_classes


