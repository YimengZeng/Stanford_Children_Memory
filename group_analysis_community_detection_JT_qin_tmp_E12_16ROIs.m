clear all; 
warning off;
rootpath = 'D:\matlab\HMM\';
addpath(genpath(fullfile(rootpath,'Scripts\GCCA_toolbox_sep21\')))
addpath(genpath(fullfile(rootpath,'Scrits\switchingFC\VB-HMM\Scripts\VB-HMM-GD-rev1')))
addpath(genpath('D:/matlab/SPM/spm8_scripts'))
%addpath(genpath('/mnt/mapricot/musk2/home/tianwenc/Toolbox/BCT/BCT_04_05_2014'));
addpath(genpath(fullfile(rootpath,'Scripts/CommunityDetection/third_party/2016_01_16_BCT')));
addpath(genpath(fullfile(rootpath,'Scripts/BayesianHiddenFactorAnalysis/privateUse/combinedVAR')))
addpath(genpath(fullfile(rootpath,'Scripts/BayesianHiddenFactorAnalysis/privateUse/VB-HMMFA-AR-v2')))
addpath(genpath(fullfile(rootpath,'Scripts/BayesianHiddenFactorAnalysis/VB-HMMFA-NBD')))
addpath(genpath(fullfile(rootpath,'Scripts/Utilities')));
addpath(genpath(fullfile(rootpath,'Scripts/CommunityDetection/scripts/')));
addpath(genpath('D:/matlab/HMM/Run/functions'))

%% ======================================================================================================
%savepath = '/mnt/mapricot/musk2/home/taghia/ColleaguesProjects/Shaozheng/encoding/trained_models_22ROIs/';
currpath = fullfile(rootpath,'Run/group_analysis'); 
savepath = fullfile(rootpath,'Run/train_models/after_outputs/');
datapath = fullfile(rootpath,'Run/Formal/');
subjectlist = 'subjectlist_performance_n24.csv';
%subjectlist = 'subjectlist_performance_n24.txt';
stateSet = {'15'}; % number of states
K = 15; 
%rep = '1'; % replication for encoding session 1 and session 2
rep = 'E1E2'; %combined two sessions with 12
% trial = 'group_model_maxstates15_adjusted_pcs';
trial = '';
filtered = 0;
sessions = {'encoding_1','encoding_2'};
%% ======================================================================================================

if filtered == 1;  ts_path = strcat('TimeSeries_16ROIs_Encoding',rep,'_wm_csf_mnvt_swaor_fl_218scans'); end
if filtered == 0;  ts_path = strcat('TimeSeries_16ROIs_Encoding',rep,'_wm_csf_mnvt_swaor_nofl_218scans'); end
% if filtered == 1;  ts_path = strcat('TimeSeries_22ROIs_Encoding',rep,'_wm_cs_mnvt_swaor_fl_218scans'); end
% if filtered == 0;  ts_path = strcat('TimeSeries_22ROIs_Encoding',rep,'_wm_cs_mnvt_swaor_nofl_218scans'); end
%if filtered == 1;  ts_path = strcat('TimeSeries_50ROIs_Encoding',rep,'_wm_cs_mnvt_swaor_fl_218scans'); end
%if filtered == 0;  ts_path = strcat('TimeSeries_50ROIs_Encoding',rep,'_wm_cs_mnvt_swaor_nofl_218scans'); end

modelpath = strcat(savepath,'model_EncodingE1E2_regMov_mean_dt1_24ss_raw_16ROIs_ss1to24_latest.mat');
% modelpath = strcat(savepath,'model_Encoding',rep,'_regMov_mean_dt1_24ss_raw_22ROIs_ss1to24.mat');
%modelpath = strcat(savepath,'train_models2/rgpca_bsds_encoding_',rep); %For 22 ROIs
%modelpath = strcat(savepath,'train_models2/rois_50/rgpca_bsds_encoding_',rep); %For 50 ROIs
%fid = fopen(fullfile(currpath,subjectlist));

%% load the memory status
memory_onset = strcat('onset_hit_miss_confidence_EncodingE1E2_n24_new.mat');
load(fullfile(datapath,ts_path,memory_onset));
memory = memory_data; clear memory_data;
%performance = textscan(fid,'%s %s %s %s %s %s %s %s %s %s %s','Delimiter','\t'); 
performance = csvread(fullfile(currpath,subjectlist));
pdata = performance(:,3:end);
%fid = fclose(fid);
% for i = 2:length(performance{1})
%     pdata(i-1,1) = str2num(performance{2}{i}); %d-prime
%     pdata(i-1,2) = str2num(performance{3}{i}); %AccuracyAll
%     pdata(i-1,3) = str2num(performance{4}{i}); %AccuracyHighest
%     pdata(i-1,4) = str2num(performance{5}{i}); %AccuracyHigh
%     pdata(i-1,5) = str2num(performance{6}{i}); %AccuracyLow
%     pdata(i-1,6) = str2num(performance{7}{i}); %FA rate
%     pdata(i-1,7) = str2num(performance{8}{i}); %Acc for encoding 1
%     pdata(i-1,8) = str2num(performance{9}{i}); %Acc for encoding 2
%     pdata(i-1,9) = str2num(performance{10}{i}); %dprime for encoding 1
%     pdata(i-1,10) = str2num(performance{11}{i}); %dprime for encoding 2    
% end
% if rep == '1'  E12_Acc = pdata(:,7); E12_dprime = pdata(:,9); end
% if rep == '2'  E12_Acc = pdata(:,8); E12_dprime = pdata(:,10); end

%% load 22/50 ROIs time series data
ts_data = strcat('16ROI_ts_0a25encoding12_DMN_n24_latest.mat');
%ts_data = strcat('22ROIs_ts_encoding_',rep,'_n24.mat');
%ts_data = strcat('50ROI_ts_encoding_',rep,'_nofl_n24.mat');
load(fullfile(datapath,ts_path,ts_data));

nSubj = length(data); display('>>Start.....');
for subj=1:nSubj
    temp = data{subj}; %For data re-organized in sherlock 
    % temp = data(subj).all_roi_ts'; %old one
    datan{subj} = preprocess(temp);
end

for i = 1:24
    counter = 0;
    tmp_memory = memory(i).onset_hit_miss(:);
    for ii = 1:length(memory(i).onset_hit_miss)
        if (tmp_memory(ii) == 1 | tmp_memory(ii) == -1)
            counter = counter + 1;
            if i == 1 data_new{i}(:,counter) = (datan{i}(:,ii) + datan{i}(:,ii+1))/2; end 
            data_new{i}(:,counter) = (datan{i}(:,ii) + datan{i}(:,ii+1) + datan{i}(:,ii+2))/3;
            memory_new{i}(counter) = tmp_memory(ii);
        end
    end
end
roi_names2 = roi_names;

%% load the trianed model
load(fullfile(modelpath,trial));
%model = group_model; 
estStatesCell = model.temporal_evolution_of_states;
transitions = model.state_transition_probabilities;

for subj = 1:length(estStatesCell)
    estStatesCell_new{subj} = estStatesCell{subj}(1:end); %leaving out the last 6 empty scans
end
estStates = cell2mat(estStatesCell);

%% Compute the occupancy rate of each state and the mean life in subject level for a given replication
% [fractional_occupancy, mean_life]  = compute_occupancy_and_mean_life(estStatesCell,15,1);
[fractional_occupancy, mean_life]  = compute_occupancy_and_mean_life_subject_wise(estStatesCell,15);
dominant_states=model.id_of_dominant_states_group_wise;

%% Visualize memory status, probability of 10 states and discrete states for each participant
figure(1)
subplot(311);
imagesc(memory(3).onset_hit_miss');
caxis([-1,1]);
colorbar('EastOutside')
subplot(312);
imagesc(model.net.hidden.QnsCell{3}'); %Visualize for the first subject
caxis([0,1]);
colorbar('EastOutside')
subplot(313);
imagesc(estStatesCell{11});
colorbar('EastOutside')

figure(2)
subplot 211
MA = nanmean(fractional_occupancy)*100;
EA = nanstd(fractional_occupancy)*100;
ccc = distinguishable_colors(15);
for s=1:K
      bar(s,MA(s),'FaceColor',ccc(s,:),'EdgeColor',ccc(s,:))
      hold on
      errorbar(s,MA(s),EA(s), 'Color',ccc(s,:) )
      hold on
end
title('occupancy rate')
MA2 = nanmean(mean_life);
EA2 = nanstd(mean_life);
subplot 212
for s=1:K
      bar(s,MA2(s),'FaceColor',ccc(s,:),'EdgeColor',ccc(s,:))
      hold on
      errorbar(s,MA2(s),EA2(s), 'Color',ccc(s,:) )
      hold on
end
title('mean life')
% save('state_occupany_rate_encoding2.mat', 'pdata','fractional_occupancy','mean_life');

%% Compute correlation between overall memory performance and dominannt states/mean life time
%z = nmi(fractional_occupancy(1:end,1), E12_Acc(1:end,1)); 
[corrE12_pr, pvalueE12_pr] = corr(re_occupancy{1, 1}(:,2), pdata(1:end,1), 'Type', 'Spearman') %Separate for Encoding 1 or 2
[corrE12_ml, pvalueE12_ml] = corr(mean_life(1:end,[15,12,9]), pdata(1:end,4),'Type','Spearman')
% [corrE12_cb, pvalueE12_cb] = corr(combined(1:end,1:5),E12_Acc(1:end,1),'Type','Spearman'); %[corrAll, pvalueAll] = corr(fractional_occupancy(:,1:5), pdata(:,1:5), 'Type', 'Pearson');
%x = E12_dprime(2:end,1); 
x = pdata(1:end,1);
y = r2_trans(:,5); xfit = min(x):0.01:max(x); 
% y = mean_life(list,[12,15,9]);

figure(3); 
for ii = 1:3%size(y,2)
    num = strcat('11',num2str(ii));
     subplot(num); a = plot(x,y(:,ii),'o'); set(a,'MarkerEdgeColor','r','MarkerFaceColor','r');
    [p,s] = polyfit(x,y(:,ii),1); 
    [yfit,dy] = polyconf(p,xfit,s,'predopt','curve','simopt','on'); 
    line(xfit,yfit,'color','k','LineWidth',1)
    hold on
    plot(xfit,yfit-dy,'k:')
    plot(xfit,yfit+dy,'k:')
    p = []; s = [];
end

%% Transition probability
% AA = transitions;               AA(:,dominant_states(end-3:end)) = [];  AA(dominant_states(end-3:end),:) = [];%Leave out states with 0
% self_transition  = diag(AA);    other_transition = sum(AA-diag(diag(AA)));

%% Frequency of each state for items later remembered or forgotten: not significant see group_analysis_community_delection_JT_qin_temp_JT.m
K1 = 3; %The first 5 most dominant states 
top_states = dominant_states(1:K1); 
SME = occupancy_SME(datan, memory, estStatesCell_new, top_states); %for the five states from two encoding combined E1E2, and plot occupancy rate as a function of memory status.
% figure(5); figure(6) %Reserved for SME of occupancy rate  

%% Compute time evolution of hidden states for the given replication
post_states = estStates;
Labels_subj = reshape(post_states,size(datan{1},2),length(datan))';

for subj=1:length(datan)
      counter = 1;
      
      temp = Labels_subj(subj,:);
      for k=1:K
          mode_states = dominant_states;
          Labels_subj(subj,find((temp==mode_states(k))==1)) = counter;
          counter = counter +1;
      end
end
figure(7);
imagesc(Labels_subj);
pf2 = gca;
set(pf2,'YDir','normal')
colormap(cccc)
c= colorbar('location','eastoutside')
ylabel(c, 'state')
ylabel('subject')
 xlabel('time instance')
 title('time evolution')
box off

%% Covariance matrix
[groupCov,~,subjCov,~]= computeDataCovarianceFromDataUsingOnlyStates(datan,estStatesCell,K);

%% community structure 
for j=1:K
      sc(j) =  sum(estStates ==j);
end
counts_post = sc/sum(sc);

%Estimates of Covariance
for j= 1:3
      k = dominant_states(j);
      est_cov = groupCov{k};
      invD = inv(diag(sqrt(diag(est_cov))));
      pearson_corr(:,:,j) = invD*est_cov*invD;
      %Partial Correlation
      inv_est_cov = inv(est_cov);
      invD = inv(diag(sqrt(diag(inv_est_cov))));
      partial_corr(:,:,j) = -invD*inv_est_cov*invD;
end
dim = size(datan{1},1);
%% Computer intra- and inter-network connectivity strength
%est_network1 = zeros(dim,dim,K); est_network2 = zeros(dim,dim,K);

MTL = pearson_corr(1:6,1:6,:); VVS = pearson_corr(7:12,7:12,:); FPN = pearson_corr(13:end,13:end,:);
mMTL = []; mVVS = []; mFPN = [];
for i = 1:K
    tmp1 = []; tmp2 = []; tmp3 = [];
    tmp1 = MTL(:,:,i); n1 = size(tmp1,2);
    tmp2 = VVS(:,:,i); n2 = size(tmp2,2);
    tmp3 = FPN(:,:,i); n3 = size(tmp3,2);
    tmpMTL = tmp1(find(tril(ones(n1,n1))));
    tmpVVS = tmp2(find(tril(ones(n2,n2))));
    tmpFPN = tmp3(find(tril(ones(n3,n3))));
    mMTL(i) = nanmean(tmpMTL(tmpMTL~=1));
    mVVS(i) = nanmean(tmpVVS(tmpVVS~=1));
    mFPN(i) = nanmean(tmpFPN(tmpFPN~=1));
end
figure(6);
subplot 311
ccc = distinguishable_colors(15);
for s=1:K
      bar(s,mMTL(s),'FaceColor',ccc(s,:),'EdgeColor',ccc(s,:))
      ylim([0 1])
      hold on
end
title('Intra-MTL connectivity');
subplot 312
ccc = distinguishable_colors(15);
for s=1:K
      bar(s,mVVS(s),'FaceColor',ccc(s,:),'EdgeColor',ccc(s,:))
      ylim([0 1])
      hold on
end
title('Intra-VVS connectivity');
subplot 313
ccc = distinguishable_colors(15);
for s=1:K
      bar(s,mFPN(s),'FaceColor',ccc(s,:),'EdgeColor',ccc(s,:))
      ylim([0 1])
      hold on
end
title('Intra-FPN connectivity');

 input_cov_matx = pearson_corr;
% input_cov_matx = partial_corr; 
 thr = 0.45;
 for i=1:3
    input_cov_matx(:,:,i)=threshold_proportional(input_cov_matx(:,:,i),thr);
 end
 input_cov_matx(input_cov_matx < 0) = 0;
 input_cov_matx(input_cov_matx > 0) = 1;

figure(72);
for k=1:3%11
%     k = mode(dominant_states(j));
%     [est_network2(:,:,k),clust_mtx_partial(:,k)] = clusters_community_detection_Newman(input_cov_matx(:,:,k));
      [est_network2(:,:,k),clust_mtx_partial(:,k)] = community_detection_newman(input_cov_matx(:,:,k));
      %ind = strcat('34',num2str(k))
      %subplot(str2num(ind))
      subplot(1,3,k)
      axis square 
      cca_plotcausality(est_network2(:,:,k),roi_names2, .4);
      axis off
end
figure(81)
for k=1:1%11
      subplot(1,1,k)
      axis square 
      imagesc(input_cov_matx(:,:,k) )
      %caxis([0,0.3]);
      set(gca, 'XTick', [0 6, 7, 10,11, 14,15, 20], 'XTickLabel',[ ] );
      set(gca, 'YTick', [0 6, 7, 10,11, 14,15, 20], 'YTickLabel',[ ] );
      colorbar('EastOutside')
    box off
end
%%
 
figure(667);
for k=1:3
      ind = strcat('32',num2str(k));
      subplot(1,3,k)
      %subplot(str2num(ind))
      axis square 
    A = input_cov_matx(:,:,k);
    A = A - diag(diag(A));
    for rep=1:100
        gamma = 1;
        tol = 1e-2;
        max_iter = 200;
        [S(:, rep), Q(rep), gamma_opt, wo] = modularity_newman(A, gamma, max_iter,tol, 'louvain');
        % [S, Q] = community_louvain(A, gamma, [], 'negative_asym');
    end
    [~,ind_max_Q] = max(Q);
    S_opt = S(:, ind_max_Q);
    Q_opt = Q(ind_max_Q);
    display([ 'optimal gamma:  ', num2str(gamma_opt),'         maximum Q:  ', num2str(Q_opt),'         number of communites:  ',  num2str(numel(unique(S_opt)))])
    
    est_network = get_network(S_opt);
    cca_plotcausality(est_network,roi_names, 0.9);
    axis off
end
 
figure(77);
for k=1:3
      ind = strcat('22',num2str(k));
      subplot(str2num(ind))
      axis square 
       A = input_cov_matx(:,:,k);
      A = A - diag(diag(A));
      imagesc(A)
     % imagesc(partial_corr(:,:,k) )
      caxis([0,0.3]);
      set(gca, 'XTick', [0 6, 7, 10,11, 14,15, 20], 'XTickLabel',[ ] );
      set(gca, 'YTick', [0 6, 7, 10,11, 14,15, 20], 'YTickLabel',[ ] );
      colorbar('EastOutside')
    box off
end

figure(99);
colormap('jet');
cccc = distinguishable_colors(16);
% load mycolor;
% BT=[];
% ST=[];
index_zero=zeros(1,3);
% ccc = distinguishable_colors(11);
% set(gca,'xtick',[1:11]);
% set(gca,'xticklabel',[1:11]);
hold on
for k=1:3
    %     ind = strcat('32',num2str(k));
    %     subplot(str2num(ind))
    % para=betweenness_wei(pearson_corr(:,:,k));
     para=strengths_und(est_network);
    %     BT = [BT bt' index_zero];
    %     ST = [ST st index_zero];
    %     axis square
    % imagesc(partial_corr(:,:,k) )
    %     caxis([0,0.3]);
        set(gca, 'XTick', [], 'XTickLabel',[ ] );
    %     set(gca, 'YTick', [0 6, 7, 10,11, 14,15, 20], 'YTickLabel',[ ] );
    for j=1:length(para)
        bar((k-1)*16+j,para(j),'FaceColor',cccc(k,:),'EdgeColor','k')
        hold on
        %     bar(bt)
    end
    bar(k*16,index_zero(1),'FaceColor',cccc(k,:),'EdgeColor',cccc(k,:))
    hold on
end

% colorbar('EastOutside')

% set(gca, 'XTick', [0 6, 7, 10,11, 14,15, 20], 'XTickLabel',[ ] );

%% Compute correlateion between trial-based memory status and probability of states
% nState = 6;
% hit = []; miss = [];
% for subj = 1:nSubj
%     memory_status = memory(subj).onset_hit_miss;
%     confidence = memory(subj).confidence_bi;
%     for n = 1:nState
%     %% Compute memory accuracy for all items within each state (on group level)
%         t = []; h = 0; m = 0; h0 = 0; m0 = 0; h1 = 0; m1 = 0; h2 = 0; m2 = 0;
%         for t = 1:length(model.estStatesCell{2})
%             if t > 1
%                 if model.estStatesCell{2}(t-1) == n & memory_status(t,1) == 1 %& confidence(t,1) >= 3
%                     h0 = h0 + 1; end
%                 if model.estStatesCell{2}(t-1) == n & memory_status(t,1) == -1 %& confidence(t,1) <=-3
%                     m0 = m0 + 1; end
%             end    
%             if model.estStatesCell{2}(t) == n & memory_status(t,1) == 1 %& confidence(t,1) >= 3
%                h1 = h1 + 1; end
%             if model.estStatesCell{2}(t) == n & memory_status(t,1) == -1 %& confidence(t,1) <= -3
%                m1 = m1 + 1; end
%             if t < length(model.estStatesCell{2})-1
%                 if model.estStatesCell{2}(t+2) == n & model.estStatesCell{2}(t+1) == n & memory_status(t,1) == 1 %& confidence(t,1) >= 3
%                    h2 = h2 + 1; end
%                 if model.estStatesCell{2}(t+2) == n & model.estStatesCell{2}(t+1) == n & memory_status(t,1) == -1 %& confidence(t,1) <= -3
%                    m2 = m2 + 1; end
%             end
%         end %priorState_hit(subj,n) = h0; priorState_miss(subj,n) = m0; currState_hit(subj,n) = h1; currState_miss(subj,n) = m1; postState_hit(subj,n) = h2; postState_miss(subj,n) = m2;     
%         priorState_hitRate(subj,n) = (h0-m0)/(h0+m0); currState_hitRate(subj,n) = (h1-m1)/(h1+m1); postState_hitRate(subj,n) = (h2-m2)/(h2+m2);
%         %% Correlation bettween state probability and memory status in a trial-by-trial manner
%         nt1 = 0; nt2 = 0; nt3 = 0; t = 0;
%         for t = 1:length(memory_status) %go through all time points
%             if memory_status(t,1) == 1 | memory_status(t,1) == -1 
%                 %display('test');
%                 if t > 1
%                     nt1 = nt1 + 1; %number of trials
%                     prior_state(nt1,1) = model.net.hidden.QnsCell{n}(t-1,1); %The state probability that is prior to the onset of each trial
%                     memory1(nt1,1) = memory(subj).onset_hit_miss(t,1); 
%                     memory1_conf(nt1,1) = memory(subj).confidence_con(t,1);
%                 end
%                 nt2 = nt2 + 1;
%                 present_state(nt2,1) = model.net.hidden.QnsCell{n}(t,1); %The state probablility that is right the onset of each trial
%                 memory2(nt2,1) = memory(subj).onset_hit_miss(t,1); 
%                 memory2_conf(nt2,1) = memory(subj).confidence_con(t,1);
%                 if t < length(memory_status)
%                     nt3 = nt3 + 1;
%                     post_state(nt3,1) = model.net.hidden.QnsCell{n}(t+1,1);
%                     memory3(nt3,1) = memory(subj).onset_hit_miss(t,1); 
%                     memory3_conf(nt3,1) = memory(subj).confidence_con(t,1);
%                 end    
%             end
%         end
%         %x = present_state; y = memory2; y(find(y == -1)) = 0;
%         %[b,dev,stats] = glmfit(x,y,'binomial','link','logit');
%         %xx = linspace(-1.5,2); 
%         %yfit = glmval(b,xx,'logit');
%         %figure:plot(x,y,'o',xx,yfit,'-');
% 
% %         [tempCorr1,pValue1] = corr(prior_state,memory1_conf,'Type','Spearman'); prior_state = []; memory1_conf = [];
% %         [tempCorr2,pValue2] = corr(present_state,memory2_conf,'Type','Spearman'); present_state = []; memory2_conf = [];
% %         [tempCorr3,pValue3] = corr(post_state,memory3_conf,'Type','Spearman'); post_state = []; memory3_conf = [];
% % 
% %         prior_state_memory_corr(subj,n) = tempCorr1;    prior_state_memory_pval(subj,n) = pValue1;
% %         present_state_memory_corr(subj,n) = tempCorr2;  present_state_memory_pval(subj,n) = pValue2;
% %         post_state_memory_corr(subj,n) = tempCorr3;     post_state_memory_pval(subj,n) = pValue3;
%     end
% end
% anova_rm(prior_state_memory_corr);
% anova_rm(present_state_memory_corr);
% anova_rm(post_state_memory_corr);

%% Major functions in this script
