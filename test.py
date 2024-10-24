from Jellyfish_Python_API.neuracle_api import DataServerThread
import time
import numpy as np
import csv


def save_matrix_to_csv(matrix, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(matrix)

def main():
    # 初始化 DataServerThread 线程
    sample_rate = 1000
    t_buffer = 8
    thread_data_server = DataServerThread(sample_rate, t_buffer)
    # 建立TCP/IP连接
    notconnect = thread_data_server.connect(hostname='127.0.0.1', port=8712)
    if notconnect:
        raise TypeError("Can't connect JellyFish, Please open the hostport ")
    else:
        # meta包还没解析好就等待
        while not thread_data_server.isReady():
            time.sleep(1)
            continue
        # 启动线程
        thread_data_server.start()
        print('Data server start')
    print("srates: ", thread_data_server.srates)
    print("maxDigital: ", thread_data_server.maxDigital)
    print("minDigital: ", thread_data_server.minDigital)
    print("channelNames: ", thread_data_server.channelNames)
    print("maxPhysical: ", thread_data_server.maxPhysical)
    print("minPhysical: ", thread_data_server.minPhysical)
    time.sleep(5)
    thread_data_server.stop()



if __name__ == '__main__':
    main()
