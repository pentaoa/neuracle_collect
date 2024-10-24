import csv
import numpy as np
import mne_integration
import pyedflib
from datetime import datetime
import matplotlib

# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

# %matplotlib qt

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


def csv_to_mne_data(csv_data, sfreq):
    data_array = np.array(csv_data).T  # Transpose the data for MNE
    # data_array = data_array / 1000000
    ch_names = [f'Channel {i + 1}' for i in range(data_array.shape[0])]
    ch_types = ['eeg'] * len(ch_names)
    print(data_array.shape[0])
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data_array, info)
    print(info)
    return raw


def convert_fif_to_edf(mne_data, output_edf_file):
    # Extract EEG data
    data, times = mne_data[:, :]

    # Get channel names and types
    channel_names = mne_data.ch_names
    channel_types = ['EEG'] * len(channel_names)

    # Create EDF file
    with pyedflib.EdfWriter(output_edf_file, len(channel_names), file_type=pyedflib.FILETYPE_EDFPLUS) as edf_writer:
        for i in range(len(channel_names)):
            edf_writer.setPhysicalMaximum(i, 32767)  # Set physical maximum
            edf_writer.setPhysicalMinimum(i, -32768)  # Set physical minimum
            edf_writer.setDigitalMaximum(i, 32767)  # Set digital maximum (16-bit resolution)
            edf_writer.setDigitalMinimum(i, -32768)  # Set digital minimum (16-bit resolution)
            edf_writer.setPhysicalDimension(i, 'uV')  # Set physical dimension
            edf_writer.setTransducer(i, '')  # Set transducer
            edf_writer.setPrefilter(i, '')  # Set prefilter
            edf_writer.setLabel(i, channel_names[i])  # Set channel label

        # Get the recording start time from the MNE data
        recording_start_time = mne_data.info['meas_date']
        if recording_start_time is None:
            # Use current time as default if recording_start_time is None
            recording_start_time = datetime.now()

        edf_writer.setStartdatetime(recording_start_time)  # Set recording start time
        edf_writer.writeSamples(data)


if __name__ == "__main__":
    # Replace 'your_file_path.csv' with the actual path to your CSV file
    csv_data = read_csv_data('./csv_data/raw2023814_1.csv')
    sfreq = 125  # Replace with the sampling frequency of your data
    mne_data = csv_to_mne_data(csv_data, sfreq)
    # mne_data.plot()
    print(mne_data.info['sfreq'])

    # Replace 'output_file.fif' and 'output_file.edf' with the desired output FIF and EDF file paths
    convert_fif_to_edf(mne_data, 'output_file.edf')

    raw_data = mne.io.read_raw_edf('output_file.edf', preload=True)
    raw_data.plot()
    print(raw_data.info['sfreq'])
    # npy_data = raw_data.get_data()

    # plt.subplot(3, 1, 1)
    # plt.plot(npy_data[0])
    # plt.subplot(3, 1, 2)
    # plt.plot(npy_data[1])
    # plt.subplot(3, 1, 3)
    # plt.plot(npy_data[16])
    # plt.show()
