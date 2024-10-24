import os.path

import numpy as np
import pandas as pd
import csv


def read_csv_data(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row if present
        for row in reader:
            lis = [float(val) for val in row[1:17]]
            lis.append(float(row[31]))
            data.append(lis)
    return data


def save_npy_files(data_array, output_dir, tasks):
    n = len(tasks)
    num = 30
    stimulate_count = [num] * n
    imagine_count = [num] * n

    for index, trigger_value in enumerate(data_array[16]):
        if trigger_value == 0:
            continue
        else:
            trigger_type = int((trigger_value - 40) / 10) % 2
            if trigger_type == 1:
                data = data_array[0:16, index - 125:index]
                trigger = int((trigger_value - 50) / 20)
                typedir = os.path.join(output_dir, f"stimulate_data")
                subdir = os.path.join(typedir, f"{tasks[trigger]}")
                os.makedirs(subdir, exist_ok=True)
                npy_file_name = os.path.join(subdir, f"{stimulate_count[trigger]}.npy")
                np.save(npy_file_name, data)
                stimulate_count[trigger] += 1
            else:
                trigger = int((trigger_value - 60) / 20)
                data = data_array[0:16, index - 375:index]
                typedir = os.path.join(output_dir, f"imagine_data")
                subdir = os.path.join(typedir, f"{tasks[trigger]}")
                os.makedirs(subdir, exist_ok=True)
                npy_file_name = os.path.join(subdir, f"{imagine_count[trigger]}.npy")
                np.save(npy_file_name, data)
                imagine_count[trigger] += 1


if __name__ == "__main__":

    # tasks1 = ['bonnet', 'cash_machine', 'coal', 'diamond', 'flower', 'headphones', 'hovercraft', 'hummingbird', 'lego',
    #           'moss']
    # tasks2 = ['bonnet', 'credit_card', 'diskette', 'goose', 'hovercraft', 'ice_pack', 'letter_opener', 'locker',
    #           'seahorse', 'sorbet']
    tasks3 = ['baton', 'boy', 'domino', 'eyeliner', 'honeycomb', 'limousine', 'mulch', 'shopping_cart', 'skin', 'torso']

    csv_file = 'csv_data/raw2023818_2.csv'
    output_directory = "new_data"
    os.makedirs(output_directory, exist_ok=True)
    csv_data = read_csv_data(csv_file)
    data_array = np.array(csv_data).T
    save_npy_files(data_array, output_directory, tasks3)
    print(np.shape(data_array))
