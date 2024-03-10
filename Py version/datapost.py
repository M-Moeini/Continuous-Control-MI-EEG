import pandas as pd
from collections import Counter
import dataprep


#Self written function which calculates and number of mismatches and return them
def custom_accuracy(y_true, y_pred):
    mismatches = []
    total = len(y_true)
    mismatch_count = 0
    
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label != pred_label:
            mismatches.append(i)
            mismatch_count += 1
            
    accuracy = 1 - (mismatch_count / total)
    
    return accuracy, mismatch_count, mismatches

#Gets majority vote with 3(can be modify with window_size) next samples (Also removes the last 2 samples)
def majority_vote_sliding_with_next(prediction_list, window_size=3):
    majority_votes = []
    
    for i in range(len(prediction_list) - window_size + 1):
        window = prediction_list[i:i+window_size]
        window_tuple = tuple(window)
        counts = Counter(window_tuple)
        majority = counts.most_common(1)[0][0]
        majority_votes.append(majority)
        
    return majority_votes

#Gets majority vote with 3(can be modify with window_size) previous samples (Also removes the first 2 samples)
def majority_vote_sliding_with_prev(prediction_list, window_size=3):
    majority_votes = []
    
    for i in range(len(prediction_list)):
        if i >= window_size - 1:
            start_index = i - window_size + 1
            window = prediction_list[start_index:i+1]
            counts = Counter(window)
            majority = counts.most_common(1)[0][0]
            majority_votes.append(majority)
        
    return majority_votes

#Gets majority vote with 3(can be modify with window_size) previous samples (Also keeps the first 2 samples as their own)
def majority_vote_sliding_with_prev_v2(prediction_list, window_size=3):
    majority_votes = []
    
    for i in range(len(prediction_list)):
        start_index = max(0, i - window_size + 1)
        window = prediction_list[start_index:i+1]
        counts = Counter(window)
        majority = counts.most_common(1)[0][0]
        majority_votes.append(majority)
        
    return majority_votes

#Gets average across participant for each task
def get_results_average(path,p_num_list,class_list):
    PATH = path
    vf = pd.DataFrame(columns=dataprep.column_names_v2) 
    for p_num in p_num_list:
        print(p_num)
        rf = pd.read_csv(PATH + "P" + str(p_num) + ".csv")
        vf = pd.concat([vf, rf], ignore_index=True)
    vf.to_csv(PATH+ 'ResultsOfAll.csv', index=False)

    columnNames = ['class','test_acc','vote_acc']
    kf = pd.DataFrame(columns=columnNames)
    kf.to_csv(PATH+'AverageAcc.csv',index=False)
    vf = pd.read_csv(PATH +"ResultsOfAll.csv")
    df = vf
    blk_list = [1234]
    for class_ in class_list:
        gf = df[(df['class1'] == class_)]
        avg_test = gf['test_acc'].mean()
        avg_vote = gf['test_acc_vote'].mean() 
        new_row = [class_, avg_test,avg_vote] 
        new_row_df = pd.DataFrame([new_row], columns=columnNames)
        rf = pd.read_csv(PATH + 'AverageAcc.csv')
        cf = pd.concat([rf, new_row_df], ignore_index=True)
        cf.to_csv(PATH +'AverageAcc.csv',index=False)  
    kf = pd.read_csv(PATH +'AverageAcc.csv') 
    print(kf.head())

