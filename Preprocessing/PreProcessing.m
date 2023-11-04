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
labels = zeros(rowNums,1);
Data = [Data,labels];
Data = num2cell(Data);
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
    Data{trigger_time,65} = 'End';
end    

%% labeling intervals
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









% for i = 1:length(triggers)
%     
%     triggers(i) = 
% 
%     
% end    



% disp(EEG.event(2).latency);
% disp(floor(EEG.event(2).latency));

%% Temp



% for i = 1:num_events
%     EEG = ChangeEventName(EEG, Names(i));
% end

% pop_saveset(EEG, ['F:\Participants\P1\Preprocessed\P1B', num2str(02)]);

% task_time = [12,8,20,16];
% rest_time = [16,12,8,20];
% 
% 
% for j=1:length(Names)
%     event_indices = find(strcmp({EEG.event.type}, Names(j)));
%     for i=1:length(event_indices)
%     
%     index = event_indices(i);
% %     value = task_time(i);
% %     string = strcat('F',num2str(8));
% %   
%     string = NamesNew(j);
%     EEG.event(index).code = string;
%     EEG.event(index).type = string;
%     
% 
%     end
%     
% end    




%     for i = 1:1
%         data = pop_epoch(EEG, {"Re"},[-1,7]);
%         data = pop_rmbase(data,[],[],[]);
%         if i == 1
%             Epochs = data;
%         else
%             Epochs = pop_mergeset(Epochs, data, 1);
%         end
%     end
% 


% % Find the column index of 'type'
% typeColumnIndex = find(strcmp(myMatrix(1,:), 'type'));
% 
% % Replace values in the 'type' column
% myMatrix(:, typeColumnIndex) = {'a'; 'b'; 'c'; 'd'};
% 
%      allEvents = {EEG.event.code};
%      
%      columnIndex = find(strcmp(EEG.event(1,:),'code'))
%      EEG.event.type = {'a'; 'b'; 'c'; 'd';'a'; 'b'; 'c'; 'd';'a'; 'b'};
%      
%      allEvents = {EEG.event.code};
%      EventsWithCurrentName = strcmp(allEvents, Name);
%      [EEG.event(EventsWithCurrentName).type] = deal(Name{:});
% disp(class(EEG.event.code))


%%Adjusting event delays
% EEG = ChangeEventLatency(EEG, delay);


% task_times_test = {[12, 16, 20, 8],
%                 [16, 12, 20, 8],
%                 [20, 16, 8, 12],
%                 [20, 12, 8, 16]};
% 
% rest_times_test = {[20, 8, 16, 12],
%     [16, 20, 8, 12],
%     [12, 20, 16, 8],
%     [20, 12, 8, 16]};

%     for i = 1:1
%         data = pop_epoch(EEG, {8},{});
%         data = pop_rmbase(data,[],[],[]);
%         if i == 1
%             Epochs = data;
%         else
%             Epochs = pop_mergeset(Epochs, data, 1);
%         end
%     end


% epoch_start = 2;   % 2000 ms = 2 seconds
% epoch_end = 2.5;   % 2500 ms = 2.5 seconds

% Create epochs based on the defined time range
% EEG = pop_epoch(EEG, {"Feet"}, [-0.1,7.9]);
% % 
% % Label the epochs as 'x'
% for i = 1:length(EEG.epoch)
%     EEG.epoch(i).eventtype = 'x';
% end
% 


% eventA_indices = find(strcmp({EEG.event.type}, 'Feet'));
% for i=2:2
%     
%     index = eventA_indices(i);
%     value = task_time(i);
%     string = strcat('F',num2str(8));
%     EEG.event(index).code = string;
%     EEG.event(index).type = string;
%     
% 
% end
% data = EEG


%     for i = 1:1
%         data = pop_epoch(EEG, {'F8'},[-1,8],'eventindices',[7]);
% %         data = pop_rmbase(data,[],[],[]);
%         if i == 1
%             Epochs = data;
%         else
%             Epochs = pop_mergeset(Epochs, data, 1);
%         end
%     end

% indexStart = 3
% indexEnd = indexStart+1
% 
% start_point = EEG.event(indexStart).latency
% end_point = EEG.event(indexEnd).latency
% 
% disp(start_point)
% Epoch = pop_epoch(EEG, {}, [start_point, end_point],"X");

% 
% eventB_indices = find(strcmp({EEG.event.type}, 'Finish'));
% 

% Loop through 'A' and 'B' events to create epochs
% for i = 1:length(eventA_indices)
%     Define start and end points for each epoch
%     start_point = EEG.event(eventA_indices(i)).latency;
%     end_point = EEG.event(eventB_indices(i)).latency;
% 
%     Create epochs between 'A' and 'B'
%     EEG = pop_epoch(EEG, {'Feet', 'Finish'}, [start_point, end_point], 'x');
% end
% disp(eventB_indices)
% disp(eventA_indices)








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