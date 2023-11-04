clear;

%% Parameters
Num_Blocks = 7;
Participant_Num = 2;
num_events = 8;
sampling_rate = 250;
sample_time_before_trigger = 0.2;
sample_point_before_trigger = sample_time_before_trigger*sampling_rate;
delay = 250;
locpath = 'H:\Apps\Matlab\toolbox\eeglab2021.1\plugins\dipfit\standard_BEM\elec\standard_1005.elc';
Block_Start = 1;
eeg = []
b = 2;
path = ['\000', num2str(b), '.vhdr'];
triggers = ["Begin","End"];
Names = ["Begin","Finish","End", "Tongue","Feet","Hand","Mis", "Rest"];
Tasks = ["Feet","Mis","Tongue","Hand","Rest"];

EEG = [];

%% Load Data
EEG = pop_loadbv('F:\Participants\P1\', path);
%% Re-referencing the data       
EEG = pop_chanedit(EEG, 'append', 63, 'changefield', {64, 'labels', 'FCz'}, 'lookup', locpath, 'setref', {'1:63', 'FCz'});
REF = struct('labels',{'FCz'},'type',{'REF'},'theta',{-89.2133},'radius',{0.095376},'X',{0.3761},'Y',{27.39},'Z',{88.668}, 'sph_theta',{89.2133},'sph_phi',{72.8323},'sph_radius',{92.8028},'urchan',{64},'ref',{''},'datachan',{0});
EEG = pop_reref(EEG, 'FP1', 'keepref', 'on', 'refloc', REF);
EEG = pop_reref(EEG,[]);

%% Resampling the data
EEG = pop_resample(EEG,sampling_rate);

%% Adjusting the event type names

for i = 1:length(Names)
    EEG = ChangeEventName(EEG, Names(i));
end

%% Reframing data matrix
Data = EEG.data';
rowNums = size(Data,1);
labels = cell(rowNums,1);
for i = 1: length(labels)
    labels{i,1} = 'NA';
end  
Data = num2cell(Data);
Data = [Data,labels];

finish_indices = find(strcmp({EEG.event.type}, 'Finish'));
%%deleting Finish event rows
Event = EEG.event;
Event(finish_indices) = [];

%% Labeling start and end point of trials
Begin_indices = find(strcmp({Event.type}, 'Begin'));
End_indices = find(strcmp({Event.type}, 'End'));
for i = 1:length(Begin_indices)
    index = Begin_indices(i);
    trigger_time  = floor(Event(index).latency);
    Data{trigger_time,65} = 'Begin';
end   

for i = 1:length(End_indices)
    index = End_indices(i);
    trigger_time  = floor(Event(index).latency);
    disp(trigger_time);
    Data{trigger_time,65} = 'End';
end    

%% :Labeling intervals
for task = 1:length(Tasks)
    label = Tasks(task);
    
    tasks_indices = find(strcmp({Event.type}, label));
    for index = 1:length(tasks_indices)
        lower_bound = floor(Event(tasks_indices(index)).latency) ;
        upper_bound = floor(Event(tasks_indices(index)+1).latency);
        

        for row = lower_bound:upper_bound-1
            Data{row,65} = label;
        end

    end  

end

%% Removing unlabeled data
unlabeled_data = find(strcmp({Data{:,65}}, 'NA'));
Data(unlabeled_data,:) = [];


%% functions
function EEG = ChangeEventName(EEG, Name)
     allEvents = {EEG.event.code};
     EventsWithCurrentName = strcmp(allEvents, Name);
     [EEG.event(EventsWithCurrentName).type] = deal(Name{:});
end

function EEG = ChangeEventLatency(EEG, delay)
    allLatencies = [EEG.event.latency];
    allLatencies = allLatencies + delay;

    for i = 2: size(EEG.event, 2)
       [EEG.event(i).latency] = deal(allLatencies(1,i));
    end
    
end