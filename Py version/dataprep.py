import mne
from mne.decoding import CSP
import logging
import numpy as np
import pandas as pd
import pymrmr
from scipy.signal import hamming
from scipy.signal import hann
from scipy.signal import blackman
from scipy.signal import kaiser
from scipy.signal import gaussian

SAMPLING_FREQ = 250
EPOCH_LENGTH = 1000
NUMBER_OF_CHANNELS = 64
NUMBER_OF_CSP_BANDS = 9
#structure of the csv file for saving results without test_vote_acc param
column_names = ['participant', 'class1', 'class2','running_time','test_acc','train_acc','test_size','train_size','train_block','test_block']

#structure of the csv file for saving results with test_vote_acc param
column_names_v2 = ['participant', 'class1', 'class2','running_time','test_acc','train_acc','test_size','train_size','train_block','test_block','test_acc_vote']

trial_order=[['Tongue','Feet','Mis','Hand'],
            ['Feet','Mis','Hand','Tongue'],
            ['Hand','Feet','Tongue','Mis'],
            ['Tongue','Mis','Hand','Feet'],
            ['Mis','Feet','Hand','Tongue'],
            ['Feet','Hand','Tongue','Mis'],
            ['Hand','Tongue','Mis','Feet'],
            ['Tongue','Feet','Mis','Hand'],
            ['Mis','Tongue','Hand','Feet']]



#Returns the task times and rest times
def get_task_rest_times(b_num):
    if b_num == 0:
        task_time = [[12, 16, 20, 8],
                    [16, 12, 20, 8],
                    [20, 16, 8, 12],
                    [20, 12, 8, 16]]
        
        rest_time = [[20, 8, 16, 12],
                    [16, 20, 8, 12],
                    [12, 20, 16, 8],
                    [20, 12, 8, 16]]
        
    elif b_num == 1:
        task_time = [[12, 8, 20, 16],
                    [16, 20, 8, 12],
                    [8, 20, 16, 12],
                    [8, 12, 20, 16]]
        
        rest_time = [[16, 12, 8, 20],
                    [8, 20, 12, 16],
                    [20, 16, 8, 12],
                    [12, 16, 20, 8]]
        
    elif b_num == 2:
        task_time = [[16, 8, 12, 20],
                    [20, 16, 12, 8],
                    [12, 20, 8, 16],
                    [8, 12, 16, 20]]
        
        rest_time = [[8, 20, 16, 12],
                    [12, 8, 20, 16],
                    [16, 12, 20, 8],
                    [8, 12, 20, 16]]
        
    elif b_num == 3:
        task_time = [[12, 16, 20, 8],
                    [16, 12, 20, 8],
                    [20, 16, 8, 12],
                    [20, 12, 8, 16]]
        
        rest_time = [[20, 8, 16, 12],
                    [16, 20, 8, 12],
                    [12, 20, 16, 8],
                    [20, 12, 8, 16]]
        
    elif b_num == 4:
        task_time = [[16, 8, 20, 12],
                    [12, 16, 8, 20],
                    [20, 8, 12, 16],
                    [8, 20, 12, 16]]
        
        rest_time = [[8, 12, 16, 20],
                    [16, 20, 12, 8],
                    [12, 16, 8, 20],
                    [20, 8, 12, 16]]
        
    elif b_num == 5:
        task_time = [[16, 12, 8, 20],
                    [20, 16, 12, 8],
                    [8, 16, 20, 12],
                    [12, 8, 16, 20]]

        rest_time = [[12, 8, 16, 20],
                    [16, 8, 20, 12],
                    [20, 12, 16, 8],
                    [8, 16, 12, 20]]
        
    elif b_num == 6:
        task_time = [[16, 8, 12, 20],
                    [20, 8, 16, 12],
                    [8, 16, 12, 20],
                    [16, 20, 12, 8]]

        rest_time = [[16, 8, 12, 20],
                    [12, 20, 8, 16],
                    [20, 16, 12, 8],
                    [8, 16, 20, 12]]     
    elif b_num ==7:
        task_time = [[12, 8, 20, 16],
                    [16, 20, 8, 12],
                    [8, 20, 16, 12],
                    [8, 12, 20, 16]]   
               
        rest_time = [[16, 12, 8, 20],
                    [8, 20, 12, 16],
                    [20, 16, 8, 12],
                    [12, 16, 20, 8]]  
    
    elif b_num == 8:
        task_time = [[16, 8, 12, 20],
                    [20, 16, 12, 8],
                    [12, 20, 8, 16],
                    [8, 12, 16, 20]]
        
        rest_time = [[8, 20, 16, 12],
                    [12, 8, 20, 16],
                    [16, 12, 20, 8],
                    [8, 12, 20, 16]]
        
    else:
        raise("Error in block number")
    

    return task_time,rest_time

#Returns the trial time 
def trial_times_genertor(task_times,rest_times):
    block_times = [item for pair in zip(task_times, rest_times) for item in pair]
    return block_times

#Extract and seperate the epochs for task class and rest class
def class_extractor(number_of_epochs, class_1, class_2, data, labels,num_channels):
    size = sum(labels[:,0] == class_1) + sum(labels[:,0] == class_2)
    Final_labels = np.zeros((size,1)).astype(int)
    dataset = np.zeros((size,num_channels, EPOCH_LENGTH))
    index = 0
    for i in range(number_of_epochs):
        if labels[i,0] == class_1 or labels[i,0] == class_2:
            dataset[index,:,:] = data[i,:,:]
            Final_labels[index,0] = labels[i,0]
            index = index + 1
        else:
            continue
            
    return dataset, Final_labels


#Calculate CSP
def calc_csp(x_train, y_train, x_test,number_of_components):
    # csp = CSP(n_components=number_of_components, reg='ledoit_wolf', log=True)
    csp = CSP(number_of_components)


    csp_fit = csp.fit(x_train, y_train)
    train_feat = csp_fit.transform(x_train)
    test_feat = csp_fit.transform(x_test)
    return train_feat, test_feat

#Extract featues with FBCSP
def feature_extractor(dataset, labels, number_of_bands, test_data,number_of_components):

    low_cutoff = 0
    
    for b in range(number_of_bands):
        logging.getLogger('mne').setLevel(logging.WARNING)
        low_cutoff += 4
        data = dataset.copy()
        data_test = test_data.copy() 
        # print(data.shape,data_test,"train test csp shape")
        # sys.exit()
        print("Frequency range: ",low_cutoff)
        filtered_data = mne.filter.filter_data(data, SAMPLING_FREQ, low_cutoff, low_cutoff + 4, verbose = False, n_jobs = 4)
        filtered_data_test = mne.filter.filter_data(test_data, SAMPLING_FREQ, low_cutoff, low_cutoff + 4, verbose = False, n_jobs = 4)

        #PCA
        # from mne.decoding import UnsupervisedSpatialFilter
        # from sklearn.decomposition import PCA, FastICA

        # pca = UnsupervisedSpatialFilter(PCA(64), average=False)
        # pca_fit = pca.fit(filtered_data)
        # filtered_data = pca_fit.transform(filtered_data)
        # filtered_data_test = pca_fit.transform(filtered_data_test)
        # train_feats = filtered_data
        # test_feats = filtered_data_test

        # filtered_data = data
        # filtered_data_test = data_test
        
        [train_feats, test_feats] = calc_csp(filtered_data, labels[:,0], filtered_data_test,number_of_components)
        if b == 0:
            train_features = train_feats
            test_features = test_feats
        else:
            train_features = np.concatenate((train_features, train_feats), axis = 1)
            test_features = np.concatenate((test_features, test_feats), axis = 1)
    
    return train_features, test_features

#Select features with mRMR algorithm
def feature_selector(train_features, labels, number_of_selected_features):
    X = pd.DataFrame(train_features)
    y = pd.DataFrame(labels)
    K = number_of_selected_features
    
    df = pd.concat([y,X], axis = 1)
    df.columns = df.columns.astype(str)
        
    selected_features = list(map(int, pymrmr.mRMR(df, 'MID', K)))
    return selected_features

#Returns the start index of the Rest and Task intervals in a trial
def get_group_start_indices(dataframe):
    group_indices = []
    current_label = None

    for idx, row in dataframe.iterrows():
        if row.iloc[-1] != current_label:
            group_indices.append(idx)
            current_label = row.iloc[-1]

    return group_indices

#Trims the trial and return it based on Begin and End triggers
def trial_cutter(data, class_1):
    df = data.copy()
    Begin_trigger = "Begin" + "_" + class_1
    End_trigger = "End" + "_" + class_1
    Begin_idx = df[df.iloc[:, -1] == Begin_trigger].index
    End_idx = df[df.iloc[:, -1] == End_trigger].index
    trial_df = df.iloc[Begin_idx[0]+1:End_idx[0],:]
    trial_df.reset_index(drop=True, inplace=True)
    trial_df.head()
    return trial_df

#Chnage the type of Begin and End triggers to String data type and add _ to end of it for further manipulation
def Begin_End_trigger_modifier(data):
    df = data.copy()
    Begin_indexes = df[df.iloc[:, -1] == 'Begin'].index
    End_indexes = df[df.iloc[:, -1] == 'End'].index
    if(len(Begin_indexes)==len(End_indexes)):
        for i in range(len(Begin_indexes)):
            index = Begin_indexes[i]+1
            val = df.iloc[index,-1]
            df.iloc[Begin_indexes[i],-1] = "Begin" + "_" + str(val)
            df.iloc[End_indexes[i],-1]   =  "End" + "_" + str(val)
    else:
        raise ValueError("Trigger seinding Exception")
    
    return df

#Remove the extra samples which has been caused by trigger sending delay!
def data_cleaner(df,class_1,class_2,tasks_time):
    class_x = class_1
    class_y = class_2
    new_df = pd.DataFrame()
    trial_df = df.copy() 
    for i in range(len(tasks_time)):
        sample_point = tasks_time[i]*SAMPLING_FREQ
        if(trial_df.iloc[sample_point+1,-1] == class_x ):
            if(i==len(tasks_time)-1):
                temp_df = trial_df.iloc[:sample_point,:]
                new_df = pd.concat([new_df, temp_df], axis=0)
                new_df.reset_index(drop=True, inplace=True)
            else:    
                temp_df = trial_df.iloc[:sample_point,:]
                next_task_idx = trial_df[trial_df.iloc[:, -1] == class_y].index
                trial_df.drop(trial_df.index[0:next_task_idx[0]], inplace=True)
                trial_df.reset_index(drop=True, inplace=True)
                new_df = pd.concat([new_df, temp_df], axis=0)
                new_df.reset_index(drop=True, inplace=True)
                class_x,class_y = class_y,class_x

    return new_df

#calculate number of the epoch based on winow length and shifting amount
def cal_epoch(df_len,sliding_len,window_len):
    number_of_epochs = int((int(df_len-window_len)/sliding_len)) +1
    return number_of_epochs

#Trim the data with the desired window and labels it and return the dataset and the labels
def data_label_attacher(cleaned_df,class_1,class_2,random_flag,class_seperator_flag,sliding_time,window_time_length,window_type,number_of_channels):
    sliding_points = int(sliding_time*SAMPLING_FREQ)
    window_time = window_time_length
    window_sample_length = window_time*SAMPLING_FREQ
    new_df_ = cleaned_df.copy()
    new_df_.drop(cleaned_df.columns[-1], axis=1, inplace=True)
    X = new_df_.to_numpy()
    X = np.transpose(X)
    number_of_epochs = cal_epoch(int(int(len(cleaned_df)/SAMPLING_FREQ)),sliding_time,window_time)
    dataset = np.zeros((number_of_epochs,number_of_channels,window_sample_length))
    labels = np.zeros((number_of_epochs,1)).astype(int)

    index = get_group_start_indices(cleaned_df)
    index.append(len(cleaned_df))
    k = 0  
    startIdx = int(k * window_sample_length)
    endIdx = int((k+1) * window_sample_length )
    l = 0
    label = 1
    for i in range(number_of_epochs):
        
        if(startIdx>=index[l] and endIdx<=index[l+1]):
            # print(startIdx,endIdx,index[l],index[l+1],"start, end, index[l], index[l+1] in if")
            slice_X = X[:, startIdx:endIdx]

            if window_type.window_name == "Kaiser":
                kaiser_window = kaiser(window_sample_length,window_type.param_value)
                slice_X *= kaiser_window

            elif window_type.window_name == "Hamming":
                hamming_window = hamming(window_sample_length)
                slice_X *= hamming_window
            
            elif window_type.window_name == "Hanning":
                hanning_window = hann(window_sample_length)
                slice_X *= hanning_window
            
            elif window_type.window_name == "Rec":
                pass

            else:
                raise ValueError("Window type is wrong!")

            dataset[i, :, :] = slice_X
            labels[i,0] = label
            # print("i is: ",i)
            # print("label is: ",label)

        else:
            
            temp = endIdx-index[l+1]
            # print(temp,endIdx,index[l+1],"temp,end,index l+1")
            slice_X = X[:, startIdx:endIdx]
            if window_type.window_name == "Kaiser":
                kaiser_window = kaiser(window_sample_length,window_type.param_value)
                slice_X *= kaiser_window

            elif window_type.window_name == "Hamming":
                hamming_window = hamming(window_sample_length)
                slice_X *= hamming_window
            
            elif window_type.window_name == "Hanning":
                hanning_window = hann(window_sample_length)
                slice_X *= hanning_window
            
            elif window_type.window_name == "Rec":
                pass
            
            else:
                raise ValueError("Window type is wrong!")
            dataset[i, :, :] = slice_X

            if(temp<window_sample_length/2):
                # print("i is: ",i)
                # print("label is: ",label)
                labels[i,0] = label
            else:
                labels[i,0] = int(not(label))
                # print("i is: ",i)
                # print("label is: ",int(not(label)))

            if(startIdx>=index[l+1]):
                l+=1
                # print(f"label changed in i = {i}")
                label = int(not(label))

                

        startIdx+=sliding_points
        endIdx+=sliding_points

    return dataset,labels


#Gathered all functions together to deliver a labeled preprocessed trial
def preprocessor(data_,class_1,class_2,tasks_time,set_type,clean_flag,sliding_time,window_time_length,window_type,number_of_channels):
    CLASS_1 = class_1
    CLASS_2 = class_2
    df = data_.copy()
    modified_df = Begin_End_trigger_modifier(df)
    trial_df = trial_cutter(modified_df,CLASS_1)
    indexes = get_group_start_indices(trial_df)

    if clean_flag:
        cleaned_df = data_cleaner(trial_df,CLASS_1,CLASS_2,tasks_time)
        final_df = cleaned_df.copy()
    else:
        final_df = trial_df.copy()

    if set_type =="TRAIN":
        random_flag = True
    elif set_type =="TEST":
        random_flag = False
    else:
        raise("Error in set type")

  
    final_data, final_labels = data_label_attacher(final_df,CLASS_1,CLASS_2,random_flag,clean_flag,sliding_time,window_time_length,window_type,number_of_channels)
    
    return final_data,final_labels


#Build a trial set for train and test dataset by concatenating preprocessed trials
def trials_set_builder(data_dict,blocks_set,set_label,class_1,class_2,clean_flag,sliding_time,window_time_length,window_type,channels_to_remove,number_of_channels):
                                   
    counter = 0

    for b_num in blocks_set:
        trial_num = trial_order[b_num].index(class_1)
        task_times,rest_times = get_task_rest_times(b_num)
        trial_times = trial_times_genertor(task_times[trial_num],rest_times[trial_num])

        # data = remove_outliers_across_channels(data_dict[b_num],10)
        # data = remove_outliers(data_dict[b_num])
        data = data_dict[b_num]
        # data = apply_median_filter(data,9)
        if class_1== 'Tongue' or class_1 == 'Mis':
            data = channel_remover(data,channels_to_remove)
            number_of_channels =  NUMBER_OF_CHANNELS-len(channels_to_remove)
        else:
            number_of_channels = NUMBER_OF_CHANNELS

        df = data.copy()
        # last_column = df.pop(df.columns[-1])
        # df.drop(df.columns[-1], axis=1, inplace=True)
        # eeg_data = df.to_numpy().T  # Transpose to have channels in columns

        # channel_names = [f'Ch{i+1}' for i in range(63)]

        # # Create MNE-Python RawArray object
        # info = mne.create_info(ch_names=channel_names, sfreq=sampling_freq, ch_types='eeg')
        # raw = mne.io.RawArray(eeg_data, info)

        # # Apply ICA
        # ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
        # ica.fit(raw)
        # ica_components = ica.get_components()

        # # Convert the ICA components to a DataFrame
        # df2 = pd.DataFrame(data=ica_components.T, columns=channel_names)
        # df2 = df2.assign(LastColumn=last_column)
        # # df = data.copy(deep=False)
        dataset,labels = preprocessor(df,class_1,class_2,trial_times,set_label,clean_flag,sliding_time,window_time_length,window_type,number_of_channels)

        if counter == 0 :
            final_data = dataset
            final_labels = labels
            # print("Before concatenation - final_data shape:", final_data.shape, "dataset shape:", dataset.shape)
        else:
            final_data = np.vstack((final_data, dataset))
            final_labels = np.vstack((final_labels, labels))
            print("After concatenation - final_data shape:", final_data.shape, "final_labels shape:", final_labels.shape)

        counter+=1 
    return final_data,final_labels



#Remove desired channels from data
def channel_remover(df, channels):
    
    df_copy = df.copy()
    df_copy.drop(df.columns[channels], axis=1, inplace=True)
    return df_copy






    
