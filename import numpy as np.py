import numpy as np
import os

data_type = [ 'passive_data', 'active_data']


# 创建基于事件的数据，生成三维数组，第一维是事件，第二维是通道，第三维是时间点
def create_event_based_npy(source_dir, preprocessed_dir, output_dir):
    subjects_list = os.listdir(source_dir)
    for subject in subjects_list:
        cls_list = os.listdir(os.path.join(source_dir, subject, data_type[1]))
        for cls in cls_list:
            cls_path = os.path.join(source_dir, subject, data_type[1], cls)
            data_list = os.listdir(cls_path)
            for data in data_list:
                data_path = os.path.join(source_dir, subject, data_type[1], cls, data)
                raw_data = np.load(data_path)
                events = raw_data[64, :]  # 第65行存储event信息

                preprocessed_data_path = os.path.join(preprocessed_dir, subject, data_type[1], cls, data)
                preprocessed_data = np.load(preprocessed_data_path)
                
                event_based_data = []
                event_indices = np.where(events > 0)[0]  # 找到所有非零的event索引
                print(event_indices)

                # 将原始数据的索引转换为降采样后的索引
                event_indices = event_indices // 4
                for idx in event_indices:
                    if idx + 25 <= preprocessed_data.shape[1]:  # 确保索引不越界
                        event_data = preprocessed_data[:64, idx:idx + 25] # 取出event前后25个时间点的数据，对应 0.1 秒
                        event_based_data.append(event_data)
                
                event_based_data = np.array(event_based_data)
                
                save_path = os.path.join(output_dir, subject, data_type[1], cls, data)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, event_based_data)


# 拼接所有数据
def concatenate_event_data(output_dir):
    concatenated_data = []
    subjects_list = os.listdir(source_dir)
    for subject in subjects_list:
        cls_list = os.listdir(os.path.join(source_dir, subject, data_type[1]))
        for cls in cls_list:
            cls_path = os.path.join(source_dir, subject, data_type[1], cls)
            data_list = os.listdir(cls_path)
            for data in data_list:
                data_path = os.path.join(output_dir, subject, data_type[1], cls, data)
                event_data = np.load(data_path)
                concatenated_data.append(event_data)
    
    # 在第一维度（事件维度）进行拼接
    concatenated_data = np.concatenate(concatenated_data, axis=0)
    
    # 保存拼接后的数据
    save_path = os.path.join(output_dir, subject, data_type[1], cls, 'concatenated_data.npy')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, concatenated_data)



source_dir = r"C:\Users\Grada\Desktop\enriched_data"
preprocessed_dir = r"C:\Users\Grada\Desktop\preprocessed_data"
output_dir = r"C:\Users\Grada\Desktop\output_data"

# create_event_based_npy(source_dir, preprocessed_dir, output_dir)
print("Event-based data saved to", output_dir)
concatenate_event_data(output_dir)
print("Event-based data concatenated and saved to", output_dir)