from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC as SVM
from sklearn.ensemble import ExtraTreesClassifier as ET
from xgboost import XGBClassifier
import numpy as np
import os
import dataprep
import datapost
import datastream
import datapredict
import dataread
from window_class import Window


path = '/home/mahdi146/projects/def-b09sdp/mahdi146/Cedar/Classification/Participants/Pickels_Participants/'
p_num_list = [9]
block_list = [0, 1, 2, 3, 4, 5, 6]
classifier_dic = {"XGB":XGBClassifier()}
number_of_components_list = [10]
number_of_selected_features_list = [10]
overlap_percent_list = [100]
channels_to_remove = []
number_of_channels = dataprep.NUMBER_OF_CHANNELS
window_time_length_list = [4]
vote_window = 3
# window_type_list_ = [Window(None,"Rec"),Window(1.5,"Kaiser"),Window(2,"Kaiser"),Window(2.5,"Kaiser"),Window(3,"Kaiser"),Window(3.5,"Kaiser")]
window_type_list_ = [Window(1.5,"Beta","Kaiser")]
# window_type_list_ = [Window(0,"No_Params","Rec")]
# train_blk_set_dic = {"12345":[0,1,2,3,4],"1234":[0,1,2,3],"123":[0,1,2],"12":[0,1],"1":[0]}
# test_blk_set_dic = {"67":[5,6],"567":[4,5,6],"4567":[3,4,5,6],"34567":[2,3,4,5,6],"234567":[1,2,3,4,5,6]}
# train_blk_set_dic = {"12":[0,1],"1":[0]}
# test_blk_set_dic = {"34567":[2,3,4,5,6],"234567":[1,2,3,4,5,6]}
train_blk_set_dic = {"12345":[0,1,2,3,4]}
test_blk_set_dic = {"67":[5,6]}
# class_1_list = ['Hand','Feet','Tongue','Mis']
class_1_list = ['Hand']
class_2 = 'Rest'
# p_num_list = [3,4,5,6,7,9,10,11,13,14]
# p_num_list = [5]
PATH = "/home/mahdi146/projects/def-b09sdp/mahdi146/Cedar/Classification/EEG/Results-v3/"

# param_grid = {
#     'max_depth': [6,7,8,9,10],
#     'learning_rate': [0.01,0.3,0.5,1],
#     'n_estimators': [100,150,200,250],
#     'subsample': [0.5,1,2],
#     'colsample_bytree': [0.5,1,2],
#     'gamma': [0,0.1,0.2,0.3,1],
#     'min_child_weight': [1,2,3,4]
# }

# param_grid = {
# 'max_depth': 6,
# 'learning_rate': 0.3,
# 'n_estimators': 100,
# 'subsample': 1.0,
# 'colsample_bytree': 1.0,
# 'gamma': 0,
# 'min_child_weight': 1
# }

data_dict_list = dataread.read_data_once(path, p_num_list, block_list)


results = datapredict.classifier(classifier_dic,
               number_of_components_list,
               number_of_selected_features_list,
               overlap_percent_list,
               channels_to_remove,
               window_time_length_list,
               window_type_list_,
               train_blk_set_dic,
               test_blk_set_dic,
               class_1_list,
               PATH,
               p_num_list,
               data_dict_list,
               number_of_channels = dataprep.NUMBER_OF_CHANNELS,
               vote_window = 3
               )
# print(results)
test_acc = results[4]
print("Test acc is : ",test_acc)