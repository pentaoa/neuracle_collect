import numpy as np

# 加载 .npy 文件
# data = np.load(r'C:\Users\Grada\Desktop\preprocessed_data\test\passive_data\1\20241012-214354-data.npy', allow_pickle=True)
# data = np.load(r'C:\Users\Grada\Desktop\enriched_data\test\passive_data\1\20241012-214354-data.npy', allow_pickle=True)
# data = np.load(r'C:\Users\Grada\Desktop\preprocessed_data\test\passive_data\1\20241124-155407-data-0.npy', allow_pickle=True)
# data = np.load(r'C:\Users\Grada\Desktop\enriched_data\test\passive_data\1\20241124-155407-data-0.npy', allow_pickle=True)
# data = np.load('/Volumes/KPan/JiahaoTest/20241124-155407-data-0.npy')
data = np.load('/Volumes/KPan/yuang/20241101-131532-data-0.npy')
# data = np.load('/Volumes/KPan/yuang/20241101-131932-data-2.npy')


# 检查加载的数据类型
print("Loaded data type:", type(data))

# 确保数据是一个 NumPy 数组
if isinstance(data, np.ndarray):
    print("Data shape:", data.shape)
    
    # 打印各个维度的含义
    if data.shape[0] == 65:
        print("\nThe shape of the data is:", data.shape)
        print("The dimensions of the data are as follows:")
        print("1. Number of channels:", data.shape[0])
        print("2. Number of time points:", data.shape[1])
    else:
        print("Unexpected data shape. Please check the data format.")
else:
    raise ValueError("Loaded .npy file is not a NumPy array.")

# 解析数据
num_channels = data.shape[0]
num_time_points = data.shape[1]

print(f"\nNumber of channels: {num_channels}")
print(f"Number of time points: {num_time_points}")


# event 通道查看
# for num in data[64]:
#     if num != 0:
#         print(num)

i=0
count=0
for num in data[64]:
    i+=1
    if num != 0:
        count +=1
        # wait=input()
        print(i)
        print(num)

print(count)