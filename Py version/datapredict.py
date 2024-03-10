
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import numpy as np
import os
import dataprep
import datapost
import datastream
from window_class import Window


def classifier(classifier_dic,
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
               ):
    
    class_2 = "Rest"
    for classifier, classifier_name in zip(classifier_dic.values(),classifier_dic.keys()):
        for number_of_components in number_of_components_list:
            for number_of_selected_features in number_of_selected_features_list:
                    for overlap_percent in overlap_percent_list:                  
                        for window_time_length in window_time_length_list:
                            sliding_time_tr = sliding_time_te = window_time_length*overlap_percent/100
                            if overlap_percent != 100:
                                window_type_list = [Window(0,"No_Param","Rec")]  
                            else:
                                window_type_list = window_type_list_

                            for window_type in window_type_list:
                                for train_blk_set,train_blk_name,test_blk_set,test_blk_name in zip(train_blk_set_dic.values(),train_blk_set_dic.keys(),test_blk_set_dic.values(),test_blk_set_dic.keys()):
                                    p = 0
                                    for p_num in p_num_list:
                                        for index, class_1 in enumerate(class_1_list):
                                            import time
                                            start_time = time.time()
                                            X_tr, Y_tr = dataprep.trials_set_builder(data_dict_list[p],train_blk_set,'TRAIN',class_1,class_2,True,sliding_time_tr,window_time_length,window_type,channels_to_remove,number_of_channels)
                                            X_te, Y_te = dataprep.trials_set_builder(data_dict_list[p],test_blk_set,'TEST',class_1,class_2,True,sliding_time_te,window_time_length,window_type,channels_to_remove,number_of_channels)


                                            print(X_tr.shape,Y_tr.shape,"train shape")
                                            print(X_te.shape,Y_te.shape,"test shape")
                                            # sys.exit()

                                            [train_features, test_features] = dataprep.feature_extractor(X_tr, Y_tr, dataprep.NUMBER_OF_CSP_BANDS, X_te,number_of_components)
                                            # [train_features, test_features] =  apply_pca(X_tr,X_te,10)
                                            print(train_features.shape, "train_features shape")
                                            print(test_features.shape, "test_features shape")
                                            selected_features = dataprep.feature_selector(train_features, Y_tr, number_of_selected_features)
                                            # selected_features = np.arange(0,train_features.shape[1],1)
                                
                                            clf = classifier
                                            runs = 1
                                            train_acc_list = []
                                            test_acc_list = []
                                            vote_acc_list = []

                                            for r in range(runs):


                                                clf.fit(train_features[:, selected_features], Y_tr[:,0])
                                                # plt = plot_feature_importance(clf.feature_importances_,selected_features)
                                                # print(selected_features,"SF")
                                                y_pr_te = clf.predict(test_features[:, selected_features])
                                                y_pr_tr = clf.predict(train_features[:,selected_features])

                                                accuracy_te = accuracy_score(Y_te, y_pr_te)
                                                test_acc_list.append(accuracy_te)

                                                print(y_pr_te)
                                                accuracy_tr = accuracy_score(Y_tr,y_pr_tr)
                                                train_acc_list.append(accuracy_tr)

                            
                                                y_pr_te_Vote = datapost.majority_vote_sliding_with_prev_v2(y_pr_te,vote_window)
                                                Y_te_Vote = datapost.majority_vote_sliding_with_prev_v2(Y_te.reshape(-1),vote_window)
                                                vote_acc, num_of_mismatches ,mismatches_list = datapost.custom_accuracy(Y_te_Vote,y_pr_te_Vote)
                                                vote_acc_list.append(vote_acc)



                                                # grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
                                                # X_train = train_features[:, selected_features]
                                                # X_test = test_features[:, selected_features]
                                                # y_train = Y_tr[:,0]
                                                # y_test = Y_te
                                                # grid_search.fit(X_train, y_train)

                                                
                                                # best_params = grid_search.best_params_
                                                # best_model = grid_search.best_estimator_

                                                
                                                # y_pred_te = best_model.predict(X_test)
                                                # y_pred_tr = best_model.predict(X_train)

                                                # accuracy_te = accuracy_score(y_test, y_pred_te)
                                                # test_acc_list.append(accuracy_te)

                                                # accuracy_tr = accuracy_score(y_train,y_pred_tr)
                                                # train_acc_list.append(accuracy_tr)

                                                # y_pr_te_Vote = majority_vote_sliding_with_prev_v2(y_pred_te,vote_window)
                                                # Y_te_Vote = majority_vote_sliding_with_prev_v2(y_test.reshape(-1),vote_window)
                                                # vote_acc, num_of_mismatches ,mismatches_list = custom_accuracy(Y_te_Vote,y_pr_te_Vote)
                                                # vote_acc_list.append(vote_acc)
                                                
        
                                            end_time = time.time()
                                            running_time = end_time-start_time
                                            participant = p_num
                                            class1 = class_1
                                            class2 = class_2
                                            running_time = running_time
                                            test_acc = np.average(test_acc_list)
                                            train_acc = np.average(train_acc_list)
                                            vote_acc = np.average(vote_acc_list)
                                            test_size = X_te.shape
                                            train_size = X_tr.shape
                                            train_block = train_blk_name
                                            test_block = test_blk_name
                                            new_row = [participant, class1, class2,running_time,test_acc,train_acc,test_size,train_size,train_block,test_block,vote_acc]


                                            path = os.path.join(
                                                PATH,
                                                classifier_name,
                                                f"{number_of_components}[8-]_CSP_Components",
                                                f"{number_of_selected_features}-Selected_Features",
                                                f"{overlap_percent}%_Overlap",
                                                f"{window_time_length}_window_time_length",
                                                f"{window_type.window_name}_Window",
                                                f"{window_type.param_name}",
                                                f"{window_type.param_value}_{window_type.param_name}",
                                                f"{train_blk_name}_Train/"
                                            )


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

                                            print(train_acc_list,"train",class_1)
                                            print(test_acc_list,"test",class_1)
                                            if (len(class_1_list)!=4):
                                                print("Task list is not fully passed!")
                                                new_row+=[Y_te, y_pr_te,Y_te_Vote, y_pr_te_Vote]
                                                return new_row
                                            else:
                                            
                                                if index == 0:
                                                    # clean_csv(path + f"P{p_num}.csv")
                                                    datastream.save_csv(new_row, path + f"P{p_num}.csv")
                                                else:                                
                                                    datastream.save_csv(new_row, path + f"P{p_num}.csv")

                                            
                                        p+=1
                                    datapost.get_results_average(path,p_num_list,class_1_list)
                                
