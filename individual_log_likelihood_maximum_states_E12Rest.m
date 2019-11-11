
%%----------------------------------------------------------------------------------------------------------
% NOTE:  in your case new data is your 22-roi rest data. 
% new_data is a cell of subjects. For each subject data is a matrix of Dim*Samples.
%%new_data = load(' YOUR 22-roir REST DATA');
clear all;
rootpath = 'D:\matlab\HMM\';
addpath(genpath('D:\matlab\HMM\Scripts\Utilities'));
currpath = fullfile(rootpath,'Run\group_analysis');
savepath = fullfile(rootpath,'Run\train_models\');
datapath = fullfile(rootpath,'Run\Formal_rest\');
output = fullfile(rootpath,'Run\figure_results\');
%datapath    = '/fs/apricot1_share6/memory_consolidation/dynamic_states/Data/Data_n24/TimeSeries_22ROIs_RS2_wm_csf_mnvt_swaor_fl/';

rep = {'1'};
skip_vols   = 5; nFrame = 180; thr = 0.4; %-0.1 (State #4: best Pearson), 0.1,0.2,0.28(State #5: best for pearson/spearman),0.3,0.4,0.5
subjectlist = 'subjectlist_performance_n24.csv';
memoryfid = fopen(fullfile(currpath,subjectlist));
nSubj = 24;
num_ROI = 16;
value = 'mean';
detrend_option = 1;
%ROI_group = 'raw';
ROI_subgroup = 'raw';
subj_subgroup = 'ss1to24';
beh_col = 10;%behavior data colomn
std_num= 4;%std used to select data

%%----------------------------------------------------------------------------------------------------------
% reading PCA model and BSDS model
% trial = 'group_model_maxstates15_adjusted_pcs';
% trial = 'subj_model_maxstates15_adjusted_pcs';
 trial = 'model_EncodingE1E2_regMov_mean_dt1_24ss_raw_16ROIs_ss1to24_latest.mat';

% savepath = '/mnt/mabloo1/apricot1_share6/memory_consolidation/dynamic_states/trained_models/';
for ss = 1:length(rep)
    modelpath{ss} = strcat(savepath,'after_outputs/');
    bsds_model{ss} = load(strcat(modelpath{ss},trial));
%    pca_model{ss} =  load(fullfile(modelpath{ss}, 'rgbpca_model.mat'));    
%     bsds_model{ss} = bsds_model{ss}.group_model;
end
% nSubj = length(bsds_model{1}.subj_model.net);
%----------StateInfo at encoding 1 & 2 in relation to memory performance---------------------------------------
target_states{1} = [15,12,9]; %stateIDs: state-3 predicting memory at encoding 1
%target_states{2} = [ 13, 12, 7, 9, 14, 2]; %stateIDs: state-13 predicting memory at encoding 2
%E12_target_states = [target_states{1}, (target_states{2}+100)]; %target_states{2}+100 for avoid overlapping stateID
E12_target_states = [target_states{1}];
%----------------------------------------------------------------------------------------------------------

%----------------------------load post-encoding resting state data
new_data = load(fullfile(datapath,'16ROI_ts_0a25resting_state_1_n24_latest.mat'));
%new_data = load(fullfile(datapath,'22ROI_ts_resting_state_2_0a1_n24.mat'));
%%---------------------------load resting data after training-----------------------------------------------
for subj = 1:length(new_data.data)
    test_data{subj} = new_data.data{1,subj}(:,skip_vols+1:end-skip_vols);
end

for i=1:24
    test_data{i}=test_data{i}';
end

%%--------------------------load memory data------------------------------
%performance = textscan(memoryfid,'%s %s %s %s %s %s %s %s %s %s %s','Delimiter','\t'); 
%fclose(memoryfid);
%for i = 2:length(performance{1})
%    pdata(i-1,1) = str2num(performance{2}{i}); %d-prime for both encoding 1 & 2
%    pdata(i-1,2) = str2num(performance{3}{i}); %AccuracyAll
%    pdata(i-1,3) = str2num(performance{4}{i}); %AccuracyHighest
%    pdata(i-1,4) = str2num(performance{5}{i}); %AccuracyHigh
%    pdata(i-1,5) = str2num(performance{6}{i}); %AccuracyLow
%    pdata(i-1,6) = str2num(performance{7}{i}); %FA rate
%    pdata(i-1,7) = str2num(performance{8}{i}); %Acc for encoding 1
%    pdata(i-1,8) = str2num(performance{9}{i}); %Acc for encoding 2
%    pdata(i-1,9) = str2num(performance{10}{i}); %dprime for encoding 1
%    pdata(i-1,10) = str2num(performance{11}{i}); %dprime for encoding 2    
% end
performance = csvread(fullfile(currpath,subjectlist));
pdata = performance(:,3:end);
% %**************************************

%load('/brain/iCAN/home/Li/SCL/mean_individual_level_rest.mat');
%load('/brain/iCAN/home/Li/SCL/mean_individual_All_scl_data_el.mat')
% str = 'Pearson';
% scl = mean_individual_level;
% [Corr_scl P_scl] = corr(scl,pdata,'type',str);
%**************************************
% E12_Acc{1} = pdata(:,7); E12_dprime{1} = pdata(:,9); %performance at encoding 1
% E12_Acc{2} = pdata(:,8); E12_dprime{2} = pdata(:,10); %performance at encoding 2

% computing log-likelihood for a given target state
counter = 0;
for ss = 1:length(rep)
    h(ss) = figure(2+ss); 
    filename = fullfile(output,strcat('Rest_Encd',num2str(ss),'Individual_state_log_likelihood_zscore.jpg'));
    for ns = 1:length(target_states{ss})
        log_likelihood = []; 
%         log_likelihood = get_log_likelihood_for_new_data_given_state(test_data, bsds_model{ss}, pca_model{ss}, target_states{ss}(ns));
        for nsubj = 1:nSubj
          %  subj_bsds_model = bsds_model{ss}.subj_model.net{nsubj};
            new_data = test_data(nsubj);
            log_likelihood = get_log_likelihood_for_new_data_given_state_individual_noPCA(test_data, bsds_model{ss}, target_states{ss}(ns));
          %  log_likelihood = [log_likelihood, get_log_likelihood_for_new_data_given_state(new_data, subj_bsds_model, pca_model{ss}, target_states{ss}(ns))];
            %individual_log_likelihood(ss).data{ns} = log_likelihood;
        end
        group_log_likelihood(ss).data{ns} = log_likelihood; 
        counter = counter + 1;
%         subplot(6,2,ns); xx = 1:length(log_likelihood{subj});
%         cc = num2cell(cool(24),2);
%         hold on;
%         for subj = 1:length(log_likelihood) 
%             hh = plot(xx, zscore(log_likelihood{subj}));
%             set(hh, 'color', [cc{subj}]);
%         end
%         title(strcat('Rest:Encd',num2str(22),'State-',num2str(target_states{ss}(ns))));
    end
    saveas(h(ss),filename);  
end
%----------------------------------------------------------------------------------------------------------
% assign state labels to each TR based on maximum likelihood values
ff = figure(6); nn = 0; pool = [];

for subj = 1:length(test_data)
   merge_log_likelihood = [];
   for ss = 1:length(rep)
       for ns = 1:length(group_log_likelihood(ss).data)
           separate_log_likelihood{ss}(ns,:) = zscore(group_log_likelihood(ss).data{ns}{subj});
       end
   end
   mergeE12_log_likelihood = [separate_log_likelihood{1}]; %merge outcomes from two models i.e. encoding 1 and 2
   pool = [pool,mergeE12_log_likelihood]; %Pool all subjects into one array
   for st = 1:size(mergeE12_log_likelihood,1)
   %       nn = nn + 1; subplot(1,3,nn); histfit(zscore(mergeE12_log_likelihood(st,:)),6);
   end
   label_thr  = zeros(length(rep),length(separate_log_likelihood{ss}));
   for ss = 1:length(rep)
       tt = 0; %temp = []; ind = []; label =[];
       for nTR = 1:length(separate_log_likelihood{ss})
           temp{ss} = separate_log_likelihood{ss}(:,nTR);
           ind(ss) = find(temp{ss} == max(temp{ss}));
           tt = find(temp{ss} == max(temp{ss}) & temp{ss} > thr); %threshold the
           label(ss,nTR) = target_states{ss}(ind(ss));
           if isempty(tt) ~= 1; 
               %ind_thr(ss) = tt; 
               label_thr(ss,nTR) = target_states{ss}(tt); 
           end %threshold the 
       end
       label_data{ss}{subj} = label(ss,:);
       label_data_thr{ss}{subj} = label_thr(ss,:);
   end
   labelE12_th = zeros(1,length(mergeE12_log_likelihood));
   for nTR = 1:length(mergeE12_log_likelihood)
       tempE12 = mergeE12_log_likelihood(:,nTR);
       indE12 = find(tempE12 == max(tempE12));
       tt2 = find(tempE12 == max(tempE12) & tempE12 > thr);%threshold 
       labelE12(nTR) = E12_target_states(indE12);
       if isempty(tt2) ~= 1; 
           labelE12_thr(nTR) = E12_target_states(tt2); 
       end 
   end
   merged_label_data{subj} = labelE12;    
   merged_label_data_thr{subj} = labelE12_thr;
end

for ii = 1:size(pool,1)
    subplot(4,5,ii); 
    histfit(pool(ii,:),20,'kernel');
end
filename = fullfile(output,strcat('Indi_Distribution_Rest_mergeE12State_log_likelihood_zscore_allSubj.jpg'));
% saveas(ff,filename);

figure(7);
for ss = 1:length(rep)
    h(ss) = figure(4+ss);
    for subj = 1:length(label_data{ss})
        counter = zeros(length(target_states{ss}),2);
        for ns = 1:length(target_states{ss})
            for nTR = 1:length(label_data{ss}{subj})
                if label_data{ss}{subj}(nTR) == target_states{ss}(ns)
                     counter(ns,1) = counter(ns,1) + 1; end
                if label_data_thr{ss}{subj}(nTR) == target_states{ss}(ns)
                     counter(ns,2) = counter(ns,2) + 1; end
            end
        end
        
        re_occupancy{ss}(subj,:) = counter(:,1)/sum(counter(:,1));
        re_occupancy_thr{ss}(subj,:) = counter(:,2)/sum(counter(:,2));
        subplot(6,4,subj);
        imagesc(label_data_thr{ss}{subj});
        colorbar('EastOutside');
        title(strcat('Rest:Ecnd',num2str(ss),'Subj-', num2str(subj))); 
    end
    filename = fullfile(output,strcat('Rest_Encd',num2str(ss),'State_log_likelihood_labelled_thr.jpg')); 
    saveas(h(ss),filename);
end
h5 = figure(8);
for subj = 1:length(merged_label_data)
    counter = zeros(length(E12_target_states),2);
    for ns = 1:length(E12_target_states)
        for nTR = 1:length(merged_label_data{subj})
            if merged_label_data{subj}(nTR) == E12_target_states(ns)   
                counter(ns,1) = counter(ns,1) + 1;          end
            if merged_label_data_thr{subj}(nTR) == E12_target_states(ns)
                counter(ns,2) = counter(ns,2) + 1;          end
        end
    end
    merged_occupancy(subj,:) = counter(:,1)/170;
    merged_occupancy_thr(subj,:) = counter(:,2)/170;
    subplot(6,4,subj);
    imagesc(merged_label_data_thr{subj});
    colorbar('EastOutside');
   title(strcat('Rest:mergedE1E2 Subj-', num2str(subj)));
end
filename = fullfile(output,'Rest_mergedE12State_log_likelihood_labelled_thr.jpg'); 
saveas(h5,filename);

h6 = figure(9); K = 17; %match the same number of bars in each figure
xx = re_occupancy; xx = re_occupancy_thr;
for ss = 1:length(rep)
    subplot(3,1,ss)
    MA{ss} = [nanmean(xx{ss})*100,zeros(1,K-size(xx{ss},2))];
    EA{ss} = [nanstd(xx{ss})*100,zeros(1,K-size(xx{ss},2))];
    ccc{ss} = distinguishable_colors(K);
    for s = 1:K%length(target_states{ss})
        bar(s,MA{ss}(s),'FaceColor',ccc{ss}(s,:),'EdgeColor',ccc{ss}(s,:))
        hold on
        errorbar(s,MA{ss}(s),EA{ss}(s), 'Color',ccc{ss}(s,:) )
        hold on
    end
end
subplot(3,1,3)
MA1 = nanmean(merged_occupancy_thr)*100;
EA1 = nanstd(merged_occupancy_thr)*100;
ccc1 = distinguishable_colors(length(E12_target_states));
for s = 1:length(E12_target_states)
    bar(s,MA1(s),'FaceColor',ccc1(s,:),'EdgeColor',ccc1(s,:))
    hold on
    errorbar(s,MA1(s),EA1(s),'Color',ccc1(s,:) )
    hold on
end
title('re_occupancy rate thr');
filename = fullfile(output,'Rest_mergedE12State_re_occupancy_thr.jpg'); 
saveas(h6, filename);
% re_occupancy_thr{1}(22,:) = []; pdata(22,:) = []; %leave out N=23 with high dprime
[corrE12_pr, pvalueE12_pr] = corr(merged_occupancy(1:end,[1,2,3,4,5]), pdata(1:end,1), 'Type', 'Spearman') %Separate for encoding 1 or 2
%[corrE12_pr, pvalueE12_pr] = corr(re_occupancy_thr{2}(2:end,:), E12_dprime{2}(2:end), 'Type', 'Spearman') %Separate for encoding 1 or 2
%[corrE12_pr, pvalueE12_pr] = corr(merged_occupancy_thr(2:end,:), (E12_Acc{1}(2:end)+E12_Acc{2}(2:end)/2), 'Type', 'Pearson')
[corrE12_pr, pvalueE12_pr] = corr(merged_occupancy_thr(1:end,[1,2,3,4,5]), pdata(1:end,1), 'Type', 'Spearman') %merged encoding 1 & 2
% [corrE12_pr, pvalueE12_pr] = corr(merged_occupancy(2:end,:),  E12_Acc{1}(2:end), 'Type', 'Pearson')
% [corrE12_pr, pvalueE12_pr] = corr(merged_occupancy(2:end,:),  E12_Acc{2}(2:end), 'Type', 'Pearson')

h9 = figure(10);
for ss = 1:length(rep)
    x = pdata(:,2); %E12_Acc{ss}(2:end,1); 
    xfit = min(x):0.01:max(x); 
    y = re_occupancy{ss}(1:end,:); 
    for ii = 1:size(y,2)
        %num = strcat('44',num2str(ii));
        if ss == 1; num = ii; end
        if ss == 2; num = 14 + ii; end
        subplot(4,5,num); a = plot(x,y(:,ii),'o'); set(a,'MarkerEdgeColor','r','MarkerFaceColor','r');
        [p,s] = polyfit(x,y(:,ii),1); 
        [yfit,dy] = polyconf(p,xfit,s,'predopt','curve','simopt','on'); 
        line(xfit,yfit,'color','k','LineWidth',1)
        hold on
        plot(xfit,yfit-dy,'k:')
        plot(xfit,yfit+dy,'k:')
        p = []; s = [];
    end
end
filename = fullfile(output,'Rest_separateE1E2_States_re_occupancy_correlation.jpg'); 
saveas(h9,filename);

h10 = figure(11)
x = pdata(2:end,1); y = merged_occupancy(2:end,:); xfit = min(x):0.01:max(x); 
for ii = 1:size(y,2)
    %num = strcat('44',num2str(ii));
    subplot(4,5,ii); a = plot(x,y(:,ii),'o'); set(a,'MarkerEdgeColor','r','MarkerFaceColor','r');
    [p,s] = polyfit(x,y(:,ii),1); 
    [yfit,dy] = polyconf(p,xfit,s,'predopt','curve','simopt','on'); 
    line(xfit,yfit,'color','k','LineWidth',1)
    hold on
    plot(xfit,yfit-dy,'k:')
    plot(xfit,yfit+dy,'k:')
    p = []; s = [];
end
filename = fullfile(output,'Rest_mergedE1E2_States_re_occupancy_correlation.jpg'); 
saveas(h10,filename);
% Outcomes from encoding 1 and 2 separately
% h2 = figure(4);
% for subj = 1:length(label_data)
%     counter = zeros(length(target_states),1);
%     for ns = 1:length(target_states)
%         for nTR = 1:length(label_data{subj})
%             if label_data{subj}(nTR) == target_states(ns)
%                 counter(ns,1) = counter(ns,1) + 1;
%             end
%         end
%     end
%     re_occupancy(subj,:) = counter/170;
%     subplot(6,4,subj);
%     imagesc(label_data{subj});
%     colorbar('EastOutside');
%     if rep == '1'   title(strcat('Rest:Encd1Subj-', num2str(subj))); end
%     if rep == '2'   title(strcat('Rest:Encd2Subj-', num2str(subj))); end
% end
% % if rep == '1' filename = fullfile(output,'Rest_Encd1State_log_likelihood_labeled.jpg');     end
% % if rep == '2' filename = fullfile(output,'Rest_Encd2State_log_likelihood_labeled.jpg');     end
% 
% [corrE12_pr, pvalueE12_pr] = corr(re_occupancy(2:end,:), E12_Acc(2:end,1), 'Type', 'Pearson') %Separate for Encoding 1 or 2
%[corrE12_pr, pvalueE12_pr] = corr(re_occupancy(2:end,:), pdata(2:end,1), 'Type', 'Pearson') %Separate for Encoding 1 or 2
display('Well Done!')








