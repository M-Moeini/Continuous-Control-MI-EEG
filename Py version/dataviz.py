from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import datastream
import dataprep
from matplotlib.patches import Rectangle, Circle
from matplotlib.font_manager import FontProperties

def plot_feature_importance(feature_importances_, selected_features, figsize=(10, 6)):
    # Get feature importance scores
    feature_importance = feature_importances_

    # Create a DataFrame to display feature numbers and their importance scores
    feature_importance_df = pd.DataFrame({'Importance': feature_importance,'Feature Number': selected_features})
    
    # Sort features by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df,"f df")

    # Plot feature importance
    plt.figure(figsize=figsize)
    sns.barplot(x='Feature Number', y='Importance', data=feature_importance_df,order=feature_importance_df['Feature Number'])
    plt.title('XGBoost - Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature Number')
    # plt.savefig(f'/home/mahdi146/projects/def-b09sdp/mahdi146/Cedar/Classification/EEG/SVGs/Feature-Importance/Hand_XGB_10F_4Rec_100%.svg', format='svg')
 
    plt.show()
    return plt


#For Plotting 1 scenario 4 tasks and 5 train sets
def one_scenario_bar_chart_plotter(PATH,
                                   classifier_name,
                                   number_of_components,
                                   number_of_selected_features,
                                   overlap_percent,
                                   window_time_length,
                                   window_type,
                                   train_blk_list):

    # Define a more visually appealing and colorblind-friendly color palette
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628']
    train_blk_set_dic = {"1": colors[0], "12": colors[1], "123": colors[2], "1234": colors[3], "12345": colors[4]}

    # Labels for the bars
    labels = ['Hand', 'Feet', 'Tongue', 'Mis']
    labels_to_x = {'Hand': 1, 'Feet': 2, 'Tongue': 3, 'Mis': 4}

    x = [labels_to_x[label] * 4 for label in labels]
    x = np.array(x)

    bar_width = 0.3
    fig, ax = plt.subplots(figsize=(14, 8))  # Adjust figure size as needed

    # Generate legend patches with the new color palette
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]

    for index, (train_blk, color) in enumerate(train_blk_set_dic.items()):
        acc_list_1 = datastream.return_acc(PATH, classifier_name, number_of_components, number_of_selected_features, overlap_percent, window_time_length, window_type, train_blk)
        blk_values = acc_list_1
        ax.bar(x + index * bar_width * 1.5, blk_values, width=bar_width, color=color)

    # Enhance font readability
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'serif'  # Consider using 'Times New Roman' or 'Arial' for academic papers

    # Refine axis for readability
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Enhance ticks for clarity
    ax.tick_params(axis='x', colors='black', direction='out', length=5, width=1)
    ax.tick_params(axis='y', colors='black', direction='out', length=5, width=1)

    # Set x ticks and labels with refined font settings
    ax.set_xticks(x+1)
    ax.set_xticklabels(labels, fontsize=16, fontweight='bold')

    # Set y ticks and labels with refined font settings and add accuracy label to y-axis
    y_ticks = np.linspace(0, 1, 11)  # Adjust range and number of ticks as needed
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{int(val * 100)}%' for val in y_ticks], fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=16)  # Adding accuracy label to the y-axis

    # Refine grid lines for better visibility
    ax.grid(axis='y', linestyle='--', color='gray', alpha=0.7, linewidth=1)
    ax.grid(axis='x', linestyle='--', color='gray', alpha=0.7, linewidth=1)

    # Add legend and title with enhanced appearance
    ax.legend(legend_patches, train_blk_set_dic.keys(), fontsize=12, title='Training Blocks', title_fontsize='14', frameon=False)
    ax.set_title('Classification of Continuous Motor Imagery Tasks With XGB and Rectangular Window', fontsize=18, fontweight='bold', family='serif')  # Adjust title

    plt.tight_layout()  # Adjust layout for better fit
    # plt.savefig(f'/home/mahdi146/projects/def-b09sdp/mahdi146/Cedar/Classification/EEG/SVGs/EEG-Ground/{classifier_name}_{number_of_selected_features}SF_{window_type.window_name}_{window_type.param_name}_{window_type.param_value}_{overlap_percent}%.jpg', dpi=300, bbox_inches='tight')

    plt.show()


#For Plotting 4 scenario 4 tasks and 5 train sets
def four_scenario_bar_chart_plotter(PATH,
                                   classifier_name,
                                   number_of_components,
                                   number_of_selected_features_list,
                                   overlap_percent,
                                   window_time_length,
                                   window_type,
                                   train_blk_list,
                                   classifier_list):

    # Labels for the bars
    labels = ['Hand', 'Feet', 'Tongue', 'Mis', ' Hand', ' Feet', ' Tongue', ' Mis']
    labels_to_x = {'Hand':1, 'Feet':2, 'Tongue':3, 'Mis':4, ' Hand':-1, ' Feet':-2, ' Tongue':-3, ' Mis':-4}


    x = [labels_to_x[label]*4 for label in labels]
    x = np.array(x)


    train_blk_set_dic = {
        "1": (165/255, 165/255, 165/255),   # Gray RGB tuple
        "12": (237/255, 125/255, 49/255),   # Orange RGB tuple 
        "123": (255/255, 192/255, 0/255),   # Yellow RGB tuple
        "1234": (68/255, 114/255, 196/255), # Blue RGB tuple
        "12345": (68/255, 114/255, 100/255) # Green RGB tuple
    }




    train_blk_list = ["12345","1234","123","12","1"]
    n = len(train_blk_list)
    m = list(range(-(n-1)//2, (n-1)//2 + 1))

    bar_width = 0.3
    fig, ax = plt.subplots(figsize=(50, 20))
    legend_labels = ['b1', 'b12', 'b123', 'b1234', 'b12345']
    colors = train_blk_set_dic.values()
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]

    for number_of_selected_features in number_of_selected_features_list:
        print(number_of_selected_features) 
        for index, (train_blk, color) in enumerate(zip(train_blk_set_dic.keys(), train_blk_set_dic.values())):
            classifier_name = "XGB"
            acc_list_1 = datastream.return_acc(PATH,classifier_name,number_of_components,number_of_selected_features,overlap_percent,window_time_length,window_type,train_blk)
            classifier_name = "LDA"
            acc_list_2 = datastream.return_acc(PATH,classifier_name,number_of_components,number_of_selected_features,overlap_percent,window_time_length,window_type,train_blk)
            blk_values = acc_list_1 + acc_list_2
            print(blk_values)
            print(acc_list_1)
            if number_of_selected_features == number_of_selected_features_list[1]:
                blk_values = [-j for j in blk_values]
            print(x.shape)  
            
            
            ax.bar(x + m[index]*bar_width*1.5, blk_values, width=bar_width, color=color)
            print(index)



    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=40)
    ax.set_yticklabels(ax.get_yticks(), fontsize=40)

    y_max = max(ax.get_yticks())
    y_ticks = [i*10/100  for i in range(-10,11)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{float(val)}%' for val in y_ticks], fontsize=20)
    ax.grid(True, linestyle='--')

    ax.legend(legend_patches, legend_labels, fontsize=20, title='Legend')
    ax.set_title(f'{window_type}_{overlap_percent}%', fontsize=40, weight='bold')






    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    # Top left (B)
    s_f = number_of_selected_features_list
    clf = classifier_list
    ax.text(xlims[0], ylims[1], f'{clf[1]}/{s_f[0]}F', horizontalalignment='left', verticalalignment='top', fontsize=40, weight='bold')

    # Top right (A)
    ax.text(xlims[1], ylims[1], f'{clf[0]}/{s_f[0]}F', horizontalalignment='right', verticalalignment='top', fontsize=40, weight='bold')

    # Bottom left (C)
    ax.text(xlims[0], ylims[0], f'{clf[1]}/{s_f[1]}F', horizontalalignment='left', verticalalignment='bottom', fontsize=40, weight='bold')

    # Bottom right (D)
    ax.text(xlims[1], ylims[0], f'{clf[0]}/{s_f[1]}F', horizontalalignment='right', verticalalignment='bottom', fontsize=40, weight='bold')

    # Show the plot
    # plt.savefig(f'/home/mahdi146/projects/def-b09sdp/mahdi146/Cedar/Classification/EEG/SVGs/Bar-Charts/{clf[0]}_{clf[1]}_{s_f[0]}F_{s_f[1]}F_{window_type}_{overlap_percent}%.svg', dpi=300, bbox_inches='tight')
    plt.show()


#without ovrlap scenario (Set the following params before running aslo must have the classifer accuracy before running)
def activity_timeline_plotter_without_overlap(b_num_list,
                              class_1,
                              Classifier,
                              window_type,
                              overlap_percent,
                              window_time,
                              y_pr_te,
                              Y_te
                              ):
    
    extra_space = 9
    cte = 4
    accumulated_times = []
    previous_time = 0
    times = []
    for b_num in b_num_list:
        trial_num = dataprep.trial_order[b_num].index(class_1)
        task_times,rest_times = dataprep.get_task_rest_times(b_num)
        trial_times = dataprep.trial_times_genertor(task_times[trial_num],rest_times[trial_num])
        times+=trial_times
    res = [cte*x/window_time for x in times]

    for i in times:
        current_time = previous_time + i
        accumulated_times.append(current_time)
        previous_time = current_time

    print(accumulated_times)
    print(res)

    fig_width = 70  # Example width
    fig_height = 70  # Example height
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Add a rectangle patch for the extra space
    # extra_space_rect = Rectangle((-extra_space, -extra_space), fig_width + 2 * extra_space, fig_height + 2 * extra_space, color=extra_space_color)
    # ax.add_patch(extra_space_rect)

    x = 0
    for i, width in enumerate(res):
        color = (30/255, 164/255, 164/255) if i % 2 == 0 else (200/255, 200/255, 200/255)
        rect = Rectangle((x, 0), width, fig_height, color=color)
        ax.add_patch(rect)
        x += width



    # Accumulatively add circles
    total_width = sum(res)  # Total width of all rectangles
    circle_size = 0.75



    for (index, i), (index_, j) in zip(enumerate(y_pr_te), enumerate(Y_te)):

        if (i == j):
            if i == 1:
                circle = Circle((4*index + circle_size , 3*fig_height/4), circle_size, color='black')
            else:
                circle = Circle((4*index + circle_size , 1*fig_height/4), circle_size, color='black')
        else:
            if i == 1:
                circle = Circle((4*index + circle_size , 3*fig_height/4), circle_size, color='red')
            else:
                circle = Circle((4*index + circle_size , 1*fig_height/4), circle_size, color='red')

        ax.add_patch(circle)



    font_prop_bold = FontProperties(weight='bold')
    offset = 1
    ax.text(-extra_space + offset, 3*fig_height/4, class_1, ha='center', va='center', fontsize=60, bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.5'), fontproperties=font_prop_bold)
    ax.text(-extra_space + offset, 1*fig_height/4, 'Rest', ha='center', va='center', fontsize=60, bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.5'), fontproperties=font_prop_bold)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-extra_space, x + extra_space)
    ax.set_ylim(-0.5*extra_space, fig_height + 1.3*extra_space)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)



    ax.set_xticks(accumulated_times)
    ax.set_xticklabels([f"{int(t)}" for t in accumulated_times], fontsize=60,y=0.04)


    # Add legend
    legend_labels = [class_1, 'Rest']
    legend_handles = [Rectangle((0, 0), 1, 1, color=(30/255, 164/255, 164/255)),
                    Rectangle((0, 0), 1, 1, color=(200/255, 200/255, 200/255))]
    font_prop = FontProperties(weight='bold', size=40)

    ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', prop=font_prop, bbox_to_anchor=(1, 1))

    # Set axis labels and title
    ax.set_xlabel('Time(s)', fontsize=60)
    ax.set_title(f'Activity Timeline For {window_type} Window,{overlap_percent}% Shifting and {Classifier} Classifier', fontsize=80, weight='bold')

    # Hide the y-axis
    ax.yaxis.set_visible(False)

    # Show plot
    plt.tight_layout()
    # fig.savefig(f'/home/mahdi146/projects/def-b09sdp/mahdi146/Cedar/Classification/EEG/SVGs/Task-classification-chart/Corrected-pics/P09_{class_1}_2blk_Test_{window_type}_{overlap_percent}%_{Classifier}.svg', format='svg', dpi=600)

    plt.show()


#with ovrlap scenario
def activity_timeline_plotter_with_overlap(b_num_list,
                              class_1,
                              Classifier,
                              window_type,
                              overlap_percent,
                              window_time,
                              y_pr_te,
                              Y_te,
                              vote_flag
                              ):


    extra_space = 9
    cte = 4
    accumulated_times = []
    previous_time = 0
    times = []
    for b_num in b_num_list:
        trial_num = dataprep.trial_order[b_num].index(class_1)
        task_times,rest_times = dataprep.get_task_rest_times(b_num)
        trial_times = dataprep.trial_times_genertor(task_times[trial_num],rest_times[trial_num])
        times+=trial_times
    res = [cte*x/window_time for x in times]

    for i in times:
        current_time = previous_time + i
        accumulated_times.append(current_time)
        previous_time = current_time

    print(accumulated_times)
    print(res)


    fig_width = 70 
    fig_height = 70 
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))


    x = 0
    for i, width in enumerate(res):
        color = (30/255, 164/255, 164/255) if i % 2 == 0 else (200/255, 200/255, 200/255)
        rect = Rectangle((x, 0), width, fig_height, color=color)
        ax.add_patch(rect)
        x += width


    # Accumulatively add circles

    total_width = sum(res)  # Total width of all rectangles
    circle_size = 0.3
    t = 0
    f = 0
    for (index, i), (index_, j) in zip(enumerate(y_pr_te), enumerate(Y_te)):
        if index >=109:

            if (i == j):
                t+=1
                if i == 1:
                    circle = Circle((6+index + circle_size , 3*fig_height/4), circle_size, color='black')
                else:
                    circle = Circle((6+index + circle_size , 1*fig_height/4), circle_size, color='black')
            else:
                f+=1
                if i == 1:
                    circle = Circle((6+index + circle_size , 3*fig_height/4), circle_size, color='red')
                else:
                    circle = Circle((6+index + circle_size , 1*fig_height/4), circle_size, color='red')

        else:
            if (i == j):
                t+=1
                if i == 1:
                    circle = Circle((3+index + circle_size , 3*fig_height/4), circle_size, color='black')
                else:
                    circle = Circle((3+index + circle_size , 1*fig_height/4), circle_size, color='black')
            else:
                f+=1
                if i == 1:
                    circle = Circle((3+index + circle_size , 3*fig_height/4), circle_size, color='red')
                else:
                    circle = Circle((3+index + circle_size , 1*fig_height/4), circle_size, color='red')

        ax.add_patch(circle)

    print(t,"true")
    print(f,"false")

    font_prop_bold = FontProperties(weight='bold')
    offset = 1
    ax.text(-extra_space + offset, 3*fig_height/4, class_1, ha='center', va='center', fontsize=60, bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.5'), fontproperties=font_prop_bold)
    ax.text(-extra_space + offset, 1*fig_height/4, 'Rest', ha='center', va='center', fontsize=60, bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.5'), fontproperties=font_prop_bold)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-extra_space, x + extra_space)
    ax.set_ylim(-0.5*extra_space, fig_height + 1.3*extra_space)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    #set times
    ax.set_xticks(accumulated_times)
    ax.set_xticklabels([f"{int(t)}" for t in accumulated_times], fontsize=60,y=0.04)


    # Add legend
    legend_labels = [class_1, 'Rest']
    legend_handles = [Rectangle((0, 0), 1, 1, color=(30/255, 164/255, 164/255)),
                    Rectangle((0, 0), 1, 1, color=(200/255, 200/255, 200/255))]
    font_prop = FontProperties(weight='bold', size=40)
    ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', prop=font_prop, bbox_to_anchor=(1, 1))

    # Set axis labels and title
    ax.set_xlabel('Time(s)', fontsize=60)
    ax.set_title(f'Activity Timeline For {window_type} Window,{overlap_percent}% Shifting({vote_flag}) and {Classifier} Classifier', fontsize=80, weight='bold')

    # Hide the y-axis
    ax.yaxis.set_visible(False)

    # Show plot
    plt.tight_layout()
    # fig.savefig(f'/home/mahdi146/projects/def-b09sdp/mahdi146/Cedar/Classification/EEG/SVGs/Task-classification-chart/Corrected-pics/P09_{class_1}_2blk_Test_{window_type}_{overlap_percent}%_{Classifier}_{vote_flag}.svg', format='svg', dpi=600)
    plt.show()

def csp_component_plotter(csp_comp_list):

    hand_acc = []
    feet_acc = []
    tongue_acc = []
    mis_acc = []
    for num_of_compnent in csp_comp_list:
        #You should chnage the path
        path = f"/home/mahdi146/projects/def-b09sdp/mahdi146/Cedar/Classification/EEG/Results-v3/XGB/{num_of_compnent}_CSP_Components/No Selected Features-Selected_Features/100%_Overlap/4_window_time_length/Kaiser_Window/Beta/1.5_Beta/1234_Train/AverageAcc.csv"
        df = pd.read_csv(path)
        hand_acc.append(df[df['class'] == 'Hand']['test_acc'].values[0])
        feet_acc.append(df[df['class'] == 'Feet']['test_acc'].values[0])
        tongue_acc.append(df[df['class'] == 'Tongue']['test_acc'].values[0])
        mis_acc.append(df[df['class'] == 'Mis']['test_acc'].values[0])




    # Plotting trend lines
    plt.plot(csp_comp_list, hand_acc, marker='o', label='Hand')
    plt.plot(csp_comp_list, feet_acc, marker='o', label='Feet')
    plt.plot(csp_comp_list, tongue_acc, marker='o', label='Tongue')
    plt.plot(csp_comp_list, mis_acc, marker='o', label='Mis')

    plt.grid(True)

    # Add labels and legend
    plt.xlabel('Number of CSP Components')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy Trend for Different Classes and CSP Components')
    plt.legend()
    # plt.savefig("/home/mahdi146/projects/def-b09sdp/mahdi146/Cedar/Classification/EEG/SVGs/CSP-component-trend/csp_trend_XGB_4Kaiser1.5B_100%_No_SF_1234Train.svg", format='svg')
    plt.show()







            