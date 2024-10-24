import numpy as np
# data_path = r'C:\Users\ncclab\PycharmProjects\CognitiveTaskSet\99_neuracle\libai-ses2\20241012-114251-data.npy'
data_path = r'C:\Users\ncclab\PycharmProjects\CognitiveTaskSet\99_neuracle\yiming\20241018-213104-data.npy'
data = np.load(data_path, allow_pickle=True)
print(data[64], data.shape)
for num in data[64]:
    if num != 0:
        print(num)