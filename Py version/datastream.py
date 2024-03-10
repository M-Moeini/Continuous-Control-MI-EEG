import dataprep
from scipy.io import loadmat
import os
import pandas as pd
import pickle
from window_class import Window

#Reads data in .mat form
def data_reader(path,p_num,block_list):
    data_dict = {}
    for b_num in block_list:
        print(b_num)
        mat = loadmat(path+'P'+str(p_num)+'B'+str(b_num)+'.mat', chars_as_strings=True, mat_dtype=True, squeeze_me=True, struct_as_record=False, verify_compressed_data_integrity=False, variable_names=None)
        df = pd.DataFrame(mat['Data'])
        data_dict[b_num] = df
    return data_dict
#Reads data in .pkl form
def pickle_reader(path, p_num, block_list):
    data_dict = {}
    
    for b_num in block_list:
        print(b_num)
        file_path = os.path.join(path, f"P{p_num}/P{p_num}B{b_num}.pkl")

        try:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)

                data_dict[b_num] = data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")

    return data_dict

#Saves results as csv in desired path(if the path doesn't exist, makes the path)
def save_csv(new_row, path):
    absolute_path = os.path.abspath(path)

    # Create the directory if it doesn't exist
    directory = os.path.dirname(absolute_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        rf = pd.read_csv(absolute_path)
    except FileNotFoundError:
        rf = pd.DataFrame(columns=dataprep.column_names_v2)

    new_row_df = pd.DataFrame([new_row], columns=dataprep.column_names_v2)
    cf = pd.concat([rf, new_row_df], ignore_index=True)
    cf.to_csv(absolute_path, index=False)

#Cleans the csv data in the mention path
def clean_csv(full_path):
    df = pd.read_csv(full_path)
    clean_df = pd.DataFrame(columns=df.columns)
    clean_df.to_csv(full_path, index=False)

#Removes the csv data in the mention path
def remove_csv(full_path):
    if os.path.exists(full_path):
        os.remove(full_path)
        print(f"File {full_path} has been removed.")
    else:
        raise(f"File {full_path} does not exist.")
    

#Returns the accuracy of desired path based on the structured built directories
def return_acc(PATH,classifier_name,number_of_components,number_of_selected_features,overlap_percent,window_time_length,window_type,train_blk_set):

    path = os.path.join(
    PATH,
    classifier_name,
    f"{number_of_components}_CSP_Components",
    f"{number_of_selected_features}-Selected_Features",
    f"{overlap_percent}%_Overlap",
    f"{window_time_length}_window_time_length",
    f"{window_type}_Window",
    f"{train_blk_set}_Train/"
    )

    data_dic = {path:pd.read_csv(path+"AverageAcc.csv")}
    acc = list(data_dic[path]['test_acc'])
    return acc

def path_cleaner(classifier_dic,
                 number_of_components_list,
                 number_of_selected_features_list,
                 overlap_percent_list,
                 channels_to_remove,
                 window_time_length_list,
                 window_type_list_,
                 train_blk_set_dic,
                 test_blk_set_dic,
                 class_1_list,
                 p_num_list,
                 PATH,
                 number_of_channels = dataprep.NUMBER_OF_CHANNELS,
                 vote_window = 3,
                 ):
    class_2 = 'Rest'
    # = {"XGB":XGBClassifier()}
    #  = [10]
    #  = [10]
    #  = [100]
    #  = []
    #  = NUMBER_OF_CHANNELS
    #  = [4]
    #  = 3
    # # window_type_list_ = [Window(None,"Rec"),Window(1.5,"Kaiser"),Window(2,"Kaiser"),Window(2.5,"Kaiser"),Window(3,"Kaiser"),Window(3.5,"Kaiser")]
    #  = [Window(1.55,"Beta","Kaiser")]
    # # window_type_list_ = [Window(0,"No_Params","Rec")]
    #  = {"12345":[0,1,2,3,4],"1234":[0,1,2,3],"123":[0,1,2],"12":[0,1],"1":[0]}
    #  = {"67":[5,6],"567":[4,5,6],"4567":[3,4,5,6],"34567":[2,3,4,5,6],"234567":[1,2,3,4,5,6]}
    # # train_blk_set_dic = {"12":[0,1],"1":[0]}
    # # test_blk_set_dic = {"34567":[2,3,4,5,6],"234567":[1,2,3,4,5,6]}
    # # train_blk_set_dic = {"12345":[0,1,2,3,4]}
    # # test_blk_set_dic = {"67":[5,6]}
    # # class_1_list = ['Hand','Feet','Tongue','Mis']
    #  = ['Hand','Feet','Tongue','Mis']
    
    #  = [3,4,5,6,7,9,10,11,13,14]
    # # p_num_list = [3]
    #  = "/home/mahdi146/projects/def-b09sdp/mahdi146/Cedar/Classification/EEG/Results-v3/"


    for classifier, classifier_name in zip(classifier_dic.values(),classifier_dic.keys()):
        for number_of_components in number_of_components_list:
            for number_of_selected_features in number_of_selected_features_list:
                    for overlap_percent in overlap_percent_list:                  
                        for window_time_length in window_time_length_list:
                            sliding_time_tr = sliding_time_te = window_time_length*overlap_percent/100
                            if overlap_percent != 100:
                                window_type_list = [Window(0,"No_Params","Rec")]  
                            else:
                                window_type_list = window_type_list_

                            for window_type in window_type_list:
                                for train_blk_set,train_blk_name,test_blk_set,test_blk_name in zip(train_blk_set_dic.values(),train_blk_set_dic.keys(),test_blk_set_dic.values(),test_blk_set_dic.keys()):
                                    p = 0
                                    for p_num in p_num_list:
                                        for index, class_1 in enumerate(class_1_list):
                                            path = os.path.join(
                                                PATH,
                                                classifier_name,
                                                f"{number_of_components}_CSP_Components",
                                                f"{number_of_selected_features}-Selected_Features",
                                                f"{overlap_percent}%_Overlap",
                                                f"{window_time_length}_window_time_length",
                                                f"{window_type.window_name}_Window",
                                                f"{window_type.param_name}",
                                                f"{window_type.param_value}_{window_type.param_name}",
                                                f"{train_blk_name}_Train/"
                                            )
                                            if index == 0:
                                                clean_csv(path + f"P{p_num}.csv")
                                                # save_csv(new_row, path + f"P{p_num}.csv")
                                            else:
                                                pass
                                                # save_csv(new_row, path + f"P{p_num}.csv")

                                            print(
                                            f"classifier_name = {classifier_name}\n"
                                            f"number_of_components = {number_of_components}\n"
                                            f"number_of_selected_features = {number_of_selected_features}\n"
                                            f"overlap_percent = {overlap_percent}\n"
                                            f"window_time_length = {window_time_length}\n"
                                            f"window_type = {window_type.window_name}\n"
                                            f"window_param_name = {window_type.param_name},window_param_value = {window_type.param_value} \n"
                                            f"train_blk_name = {train_blk_name}\n"
                                            f"Participant = {p_num}"
                                            )

                                            # print(train_acc_list,"train",class_1)
                                            # print(test_acc_list,"test",class_1)

                                            
                                        p+=1
                                    # get_results_average(path,p_num_list,class_1_list)
                                    if os.path.exists(path+"AverageAcc.csv"):
                                        remove_csv(path+"AverageAcc.csv")
                                        remove_csv(path+"ResultsOfAll.csv")
                                    else:
                                        pass
                                        # get_results_average(path,p_num_list,class_1_list)
                                    