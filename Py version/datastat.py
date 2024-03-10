from collections import Counter
import numpy as np

# Find same samples in the data
def find_duplicates(data_list):
    counted_values = Counter(data_list)
    duplicate_values = {value: count for value, count in counted_values.items() if count > 1}
    return duplicate_values

#Calculates Mean, STD, and VAR for each channel and each block
def Statistical_analysor(p_num_list,data_dicts_list,num_channels):

    with open('/home/mahdi146/projects/def-b09sdp/mahdi146/Cedar/Classification/EEG/Classification/Statistics.txt', 'w') as file:
        for p in range(len(p_num_list)):
            file.write(f'Particpant: {p+3} '+'\n')
            for b in range(7):
                file.write(f'Block: {b+1} '+'\n')
                data_pd = data_dicts_list[p][b]
                data = data_pd.iloc[:, :-1]
                data_np = data.values
                eeg_data = data_np
                print("Data type:", type(eeg_data))
                print("Shape:", eeg_data.shape)
                eeg_data = np.array(eeg_data)
                mean_values = np.mean(eeg_data, axis=0)
                variance_values = np.var(eeg_data, axis=0)
                std_deviation_values = []
                
                for i in range(num_channels):
                    print(f"Channel {i + 1}:")
                    print(f"Mean: {mean_values[i]}")
                    print(f"Variance: {variance_values[i]}")
                    std_deviation_values.append(np.sqrt(variance_values[i]))
                    print(f"Standard Deviation: {std_deviation_values[i]}")
                    print()
                    file.write(f'Channel {i+1}: '+'\n')
                    file.write(f"Mean: {mean_values[i]}"+"\n")
                    file.write(f"Variance: {variance_values[i]}"+"\n")
                    file.write(f"Standard Deviation: {std_deviation_values[i]}"+"\n\n")
                
                lists_to_check = {
                'mean_values': mean_values,
                'variance_values': variance_values,
                'std_deviation_values': std_deviation_values
                }
                for list_name, data_list in lists_to_check.items():
                    duplicate_values = find_duplicates(data_list)
                    if duplicate_values:
                        print(f"Duplicate values and their counts for {list_name}:")
                        file.write(f"Duplicate values and their counts for {list_name}:"+"\n")
                        for value, count in duplicate_values.items():
                            print(f"Value: {value}, Count: {count}")
                            file.write(f"Value: {value}, Count: {count}"+"\n")
                    else:
                        print(f"No duplicate values found in the {list_name} list.")
                        file.write(f"No duplicate values found in the {list_name} list."+"\n")
