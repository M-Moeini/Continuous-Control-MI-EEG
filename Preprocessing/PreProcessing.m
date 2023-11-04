clear;

%% Parameters
Num_Blocks = 7;
Participant_Num = 2;
num_events = 8;
freq = 250;
delay = 250;
locpath = 'H:\Apps\Matlab\toolbox\eeglab2021.1\plugins\dipfit\standard_BEM\elec\standard_1005.elc';
Block_Start = 1;
eeg = []
b = 2;
path = ['\000', num2str(b), '.vhdr'];
% path2 = '0003.vhdr'
% Data = [];
EEG = [];

%% Load Data
% data1 = pop_loadbv('E:\Master\Terms\Term3\Theisi\Data\Recordings\Phase 1\Raw Data\P2\', path2);

EEG = pop_loadbv('F:\Participants\P1\', path);
% % EEG = pop_chanedit(EEG,'append',63,'changefield', {64, 'labels', 'FCz'},'lookup', locpath, 'setref', {'1:63', 'FCz'});
% %     REF = struct('labels',{'FCz'},'type',{'REF'},'theta',{-89.2133},'radius',{0.095376},'X',{0.3761},'Y',{27.39},'Z',{88.668}, 'sph_theta',{89.2133},'sph_phi',{72.8323},'sph_radius',{92.8028},'urchan',{63},'ref',{''},'datachan',{0});
% %     EEG = pop_reref(EEG, 'FP1', 'keepref', 'on', 'refloc', REF);
%% Re-referencing the data       
EEG = pop_chanedit(EEG, 'append', 63, 'changefield', {64, 'labels', 'FCz'}, 'lookup', locpath, 'setref', {'1:63', 'FCz'});
REF = struct('labels',{'FCz'},'type',{'REF'},'theta',{-89.2133},'radius',{0.095376},'X',{0.3761},'Y',{27.39},'Z',{88.668}, 'sph_theta',{89.2133},'sph_phi',{72.8323},'sph_radius',{92.8028},'urchan',{64},'ref',{''},'datachan',{0});
EEG = pop_reref(EEG, 'FP1', 'keepref', 'on', 'refloc', REF);
EEG = pop_reref(EEG,[]);

%% Resampling the data
EEG = pop_resample(EEG,freq);

%% Adjusting the event type names
Names = ["Begin","Finish","End", "Tongue","Feet","Hand","Mis", "Rest"];
NamesNew = ["Bi","Fi","En","To","Fe","Ha","Si","Re"];

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


%% Adjusting event delays
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


epoch_start = 2;   % 2000 ms = 2 seconds
epoch_end = 2.5;   % 2500 ms = 2.5 seconds

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






%% Extracting epochs, removing baseline, and merging the classes


%     for i = 1:1
% %         data = pop_epoch(EEG, {Names(1)},[-0.494 3.994]);
%         data = pop_epoch(EEG, events, 'eventtype', 'boundary', 'durations', durations);
%         data = pop_rmbase(data,[],[],[]);
%         if i == 1
%             Epochs = data;
%         else
%             Epochs = pop_mergeset(Epochs, data, 1);
%         end
%     end
%     pop_saveset(Epochs, ['E:\Master\Terms\Term3\Theisi\me\DataPlaying\P2\PTB - 2\Test2\output\T', num2str(02)]);






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