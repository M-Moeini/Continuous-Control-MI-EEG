import dataprep
import numpy as np

#Count the extra samples in a very raw block which has been caused by trigger sending delay!
def extra_samples_block_counter(df,trial_order,b_num):

    df.drop(df[df.iloc[:,-1].isin(['Begin', 'End'])].index, inplace=True)
    df.reset_index(drop=True, inplace=True)    
    df['group'] = (df.iloc[:,-1] != df.iloc[:,-1].shift(1)).cumsum()

    
    group_counts_Rest = df[df.iloc[:,-1] == 'Rest'].groupby('group').size()
    with open('sampleList.txt', 'a') as file:
        file.write(f'block {b_num+1} '+'\n')
        for j in range (len(trial_order)):
            print(trial_order[j])
            trial_num = j
            task_times,rest_times = dataprep.get_task_rest_times(b_num)
            trial_times = dataprep.trial_times_genertor(task_times[trial_num],rest_times[trial_num])
            trial_samples = [item*dataprep.SAMPLING_FREQ for item in trial_times]
            group_counts_task = df[df.iloc[:,-1] == trial_order[j]].groupby('group').size()
            sampleList = []
            for i in range(4):
                task = group_counts_task.iloc[i]
                rest = group_counts_Rest.iloc[4*j+i]
                sampleList.append(task)
                sampleList.append(rest)
            # extra_samples = [x-y for x,y in zip(sampleList,trial_samples)]
            file.write(', '.join(map(str, sampleList)) + f' trial={trial_order[j]} '+'\n')
            print(sampleList)
        file.write('\n\n')


#Shuffles the data with labels(Doesn't shuffles the labels)
def shuffler(dataset,labels):
    print(dataset.shape)
    print(labels.shape)
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    shuffled_dataset = dataset[indices]
    shuffled_labels = labels[indices]
    return shuffled_dataset,shuffled_labels



