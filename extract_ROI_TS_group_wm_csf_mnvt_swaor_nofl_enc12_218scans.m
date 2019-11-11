
% This is the configuration template file for copy data from a group of participants 
clear all;
% -------------- Please specify the individualstats server: resting data, adults and kids, 5 HTT ----------------
% paralist.stats_path     = '/fs/musk2';
paralist.raw_server     = '/brain/iCAN/data';
paralist.parent_folder  = [''];
paralist.data_type      = 'nii';
%paralist.imagefilter   = 'swar';
paralist.imagefilter    = 'swcar'; 

TR_val = 2;
bandpass_on             = 0;     
fl                      = 0.008;
% Upper frequency bound for filtering (in Hz)
fh                      = 0.25;

% Please specify file name holding subjects to be analyzed
% For one subject list files. For eg.,
paralist.subjlist_file  = {'sublist_use.txt'}; 
%paralist.subjlist_file = {'14-1-11.2_3T2'};'14-10-19.1_3T2', 
% paralist.subjlist_file= {'14-02-17.1_3T2'};
paralist.ROI_dir        = '/brain/iCAN/home/Zeng/HMM/Config/ROI_mats';
%paralist.ROI_dir        = '/mnt/mabloo1/apricot1_share6/memory_consolidation/dynamic_states/ROIs/50ROIs_n24'
%paralist.output_folder = '/mnt/mabloo1/apricot1_share6/memory_consolidation/dynamic_states/Data/Data_n24/TimeSeries_RS1';
%paralist.output_folder = '/mnt/mabloo1/apricot1_share6/memory_consolidation/dynamic_states/Data/Data_n24/TimeSeries_RS1_wm_cs_mnvt_swar';
paralist.output_folder  = '/brain/iCAN/home/Zeng/HMM/Outputs/';
%paralist.output_folder  = '/mnt/mabloo1/apricot1_share6/memory_consolidation/dynamic_states/Data/Data_n24/TimeSeries_50ROIs_Encoding1_wm_cs_mnvt_swaor_nofl_218scans';
% get roiname:
niidir = dir(paralist.ROI_dir);
niidir = niidir(3:end);
niidir = struct2cell(niidir);
niidir = niidir(1,:);
niidir = niidir';
paralist.roi_list  = niidir;
%paralist.roi_list      = {'list_18ROIs_n24_new.txt'};
%paralist.roi_list       = {'list_50ROIs_n24_new.txt'};
% ----- Please specify the session name: adults, kids, 5 HTT ----------------
paralist.non_year_dir   = [''];
paralist.sess_folder    = 'EL';
paralist.preprocessed_folder = 'smoothed_spm8';

%-white matter and CSF roi files
wm_csf_roi_file = cell(2,1);
%-white matter and csf rois
wm_csf_roi_file{1} = '/brain/iCAN/home/Zeng/SPM/spm8_scripts/rsFC_network/white_mask_p08_d1_e1_roi.mat';
wm_csf_roi_file{2} = '/brain/iCAN/home/Zeng/SPM/spm8_scripts/rsFC_network/csf_mask_p08_d1_e1_roi.mat';   
% Show the system information and write log files
warning('off', 'MATLAB:FINITE:obsoleteFunction') 
c     = fix(clock);
disp('==================================================================');
%fprintf('copy files start at %d/%02d/%02d %02d:%02d:%02d\n',c);
disp('==================================================================');
%fname = sprintf('copy and rename files -%d_%02d_%02d-%02d_%02d_%02.0f.log',c);
%diary(fname);
disp(['Current directory is: ',pwd]);
disp('------------------------------------------------------------------');

currentdir = pwd;

% -------------------------------------------------------------------------
% Read in parameters
% -------------------------------------------------------------------------
% stats_path       = strtrim(paralist.stats_path);
raw_server       = strtrim(paralist.raw_server);
%stats_folder     = strtrim(paralist.stats_folder);
parent_folder    = strtrim(paralist.parent_folder);
subjlist_file    = strtrim(paralist.subjlist_file);
output_folder    = strtrim(paralist.output_folder);
sess_folder      = strtrim(paralist.sess_folder);
data_type        = strtrim(paralist.data_type);
imagefilter      = strtrim(paralist.imagefilter);
ROI_dir          = strtrim(paralist.ROI_dir);
roi_list         = strtrim(paralist.roi_list);
non_year_dir     = strtrim(paralist.non_year_dir);
preprocessed_folder = strtrim(paralist.preprocessed_folder);
current = pwd;
roi_list         = strcat(ROI_dir, '/', roi_list);
% -------------------------------------------------------------------------
% Load subject list, constrast file and batchfile
% -------------------------------------------------------------------------
subjects        = ReadList(subjlist_file);
parent_folder   = ReadList(parent_folder);
numsubj         = length(subjects);
imagedir        = cell(numsubj,1);

%movement states
mvmntdir = cell(numsubj,2);

run_FC_dir = pwd;
% Create local folder holding temporary data
temp_dir = fullfile(run_FC_dir, 'temp1');
if exist(temp_dir,'dir')
  unix(sprintf('/bin/rm -rf %s', temp_dir));
end

%-Update path parameters
if isempty(non_year_dir)
  for i = 1:numsubj
    pfolder{i} = ['20', subjects{i}(1:2)];
  end
else
  for i = 1:sublength
    pfolder{i} = non_year_dir;
  end
end
%-update roi list
if ~isempty(roi_list)
    ROIName = ReadList(roi_list);
    NumROI = length(ROIName);
    roi_file = cell(NumROI, 1);
    for iROI = 1:NumROI
      ROIFile = spm_select('List', ROI_dir, ['^', ROIName{iROI}]);
        if isempty(ROIFile) 
          error('Folder contains no ROIs'); 
        end
        roi_file{iROI} = fullfile(ROI_dir, ROIFile);
    end
end 

for cnt = 1:numsubj
    %sessionlink_dir{cnt} = fullfile(raw_server, pfolder{cnt}, ....
    %                                subjects{cnt}, 'fmri', sess_folder);
    imagedir{cnt} = fullfile(raw_server, pfolder{cnt}, subjects{cnt}, ...
                              'fmri', sess_folder, preprocessed_folder);  
    mvmntdir{cnt,1} = fullfile(raw_server, pfolder{cnt}, subjects{cnt}, ...
                               'fmri', sess_folder, 'unnormalized');
    mvmntdir{cnt,2} = fullfile(raw_server, pfolder{cnt}, subjects{cnt}, ...
                               'fmri', sess_folder, preprocessed_folder); %                           
end
temp = num2str(fh);
%all_output = strcat(output_folder, '/50ROI_ts_0a', temp(3:end), sess_folder, '_n24', '.mat');
all_output = strcat(output_folder, '/16ROI_ts_0a', temp(3:end), sess_folder, '_n24', '.mat');
non = 0; 
for FCi = 1:numsubj
    secondpart = subjects{FCi}(1:2);
    if str2double(secondpart) > 96
        pfolder = ['19' secondpart];
    else
        pfolder = ['20' secondpart];
    end
    %fprintf('Subject %s:\n',subjects{subcnt});
    %outputfile = strcat(output_folder, '/ROI_ts_', subjects{FCi}, '_', sess_folder, '.mat');
    outputfile = strcat(output_folder, '/ROI_ts_', subjects{FCi}, '_', sess_folder, '_nofl.mat');
 
    disp('----------------------------------------------------------------');
    fprintf('Processing subject: %s \n', subjects{FCi});
    if exist(temp_dir, 'dir')
        unix(sprintf('/bin/rm -rf %s', temp_dir));
    end
    mkdir(temp_dir);
    cd(imagedir{FCi});
    
    fprintf('Copy files from: %s \n', pwd);
    fprintf('to: %s \n', temp_dir);
    if strcmp(data_type, 'nii')
        unix(sprintf('/bin/cp -af %s %s', [imagefilter, 'I.nii*'], temp_dir));
        if exist('unused', 'dir')
        unix(sprintf('/bin/cp -af %s %s', fullfile('unused', [imagefilter, 'I.nii*']), temp_dir));
        end
    else
        unix(sprintf('/bin/cp -af %s %s', [imagefilter, 'I.img*'], temp_dir));
        unix(sprintf('/bin/cp -af %s %s', [imagefilter, 'I.hdr*'], temp_dir));
        if exist('unused', 'dir')
          unix(sprintf('/bin/cp -af %s %s', fullfile('unused', [imagefilter, 'I.img*']), temp_dir));
          unix(sprintf('/bin/cp -af %s %s', fullfile('unused', [imagefilter, 'I.hdr*']), temp_dir));
        end
    end
    cd(temp_dir);
    unix('gunzip -fq *');
    newimagefilter = imagefilter;
  
    %-Bandpass filter for whole brain data if it is set to 'ON'
%     if bandpass_on == 1
%         disp('Bandpass filtering data ......................................');
%         bandpass_final_SPM(2, fl, fh, temp_dir, imagefilter, data_type);
%         disp('Done');
%         %-Prefix update for filtered data
%         newimagefilter = ['filtered', imagefilter];
%     end
     %-Step 1 ----------------------------------------------------------------
     %-Extract ROI timeseries
    disp('Extracting ROI timeseries ......................................');
    [all_roi_ts, roi_names] = extract_ROI_timeseries(roi_file, temp_dir, 1, ...
                                      0, newimagefilter, data_type);     
    all_roi_ts = all_roi_ts';
    % Total number of ROIs
    numroi = length(roi_names);
  
    %-Step 2 ----------------------------------------------------------------
    %-Extract white matter and CSF signals
    disp('Extract white matter and CSF signals ...........................');
    [wm_csf_ts, wm_csf_roi_name] = extract_ROI_timeseries(wm_csf_roi_file, temp_dir, 1, ...
                                      0, newimagefilter, data_type); 
    wm_csf_ts = wm_csf_ts';    
    
    %-Truncate ROI and global timeseries
    %all_roi_ts = all_roi_ts(NUMTRUNC(1)+1:end-NUMTRUNC(2), :);
    %all_roi_ts = all_roi_ts(NUMTRUNC(1)+1:NFRAMES-NUMTRUNC(2), :);
    %global_ts = org_global_ts(NUMTRUNC(1)+1:end-NUMTRUNC(2));
    
    %wm_csf_ts = wm_csf_ts(NUMTRUNC(1)+1:end-NUMTRUNC(2), :);
    %wm_csf_ts = wm_csf_ts(NUMTRUNC(1)+1:NFRAMES-NUMTRUNC(2), :);
    %NumVolsKept = size(wm_csf_ts, 1);
    %wm_csf_ts = wm_csf_ts - repmat(mean(wm_csf_ts, 1), NumVolsKept, 1);
   
    %===============================================================
    %-STEP 3 --------------------------------------------------------------
    %-Filtering out wm_csf_ts and mvnt
    
    %-Run through multiple ROIs
   for roicnt = 1:numroi                              
    rts = all_roi_ts(:,roicnt);
    
    %-Extract covariates for each ROI
    disp('Regressing out wm, csf, and movement signals ...........');
    unix(sprintf('gunzip -fq %s', fullfile(mvmntdir{FCi,1}, 'rp_I*')));
    unix(sprintf('gunzip -fq %s', fullfile(mvmntdir{FCi,2}, 'rp_I*')));
    rp2 = dir(fullfile(mvmntdir{FCi,2}, 'rp_I*.txt'));
    rp1 = dir(fullfile(mvmntdir{FCi,1}, 'rp_I*.txt'));
    if ~isempty(rp2)
      mvmnt = load(fullfile(mvmntdir{FCi,2}, rp2(1).name));
    elseif ~isempty(rp1)
      mvmnt = load(fullfile(mvmntdir{FCi,1}, rp1(1).name));
    else
      fprintf('Cannot find the movement file: %s \n', subjects{FCi});
      cd(current_dir);
      diary off; return;
    end

    %-Demeaned ROI timeseries and wm+csf signals
    %rts = rts - mean(rts)*ones(size(rts, 1), 1);
    
    %bandpass filtering very low and high frequency bands
    if bandpass_on == 1
        rts = bandpass_final_SPM_ts(TR_val, fl, fh, rts);
    end

    %bandpass filtering the movement parameters
    if bandpass_on == 1
        mvmnt = bandpass_final_SPM_ts(TR_val, fl, fh, mvmnt);
    end
    %mvmnt = mvmnt(NUMTRUNC(1)+1:NUMTRUNC(1)+NumVolsKept, :);
    %mvmnt = mvmnt(NUMTRUNC(1)+1:NFRAMES-NUMTRUNC(2), :);
    
    %-ts with Covariates
    %ts  = [rts wm_csf_ts mvmnt];
     %Regressor for intercept
     
    xm = ones(length(rts),1);
    xm = xm./norm(xm);
    %D = [xm global_signal csf_signal(:) wm_signal(:)];
    D = [xm wm_csf_ts mvmnt];
    
    %-Regressed out covariates of no interest
    %x = down_sample_data(:,r);
    %beta_hat = D\x;
    %x = x - (D*beta_hat);
    beta_hat = D\rts;
    rts = rts - (D*beta_hat);
    all_roi_ts(:,roicnt) = rts;
   end
   %data(FCi).all_roi_ts = all_roi_ts;
   data{FCi} = all_roi_ts';
   %data(FCi).roi_names  = roi_names;
   save(outputfile, 'all_roi_ts', 'roi_names');
end
for ii = 1:length(roi_names)
    roi_names{ii} = roi_names{ii}(5:end)';
end
save(all_output, 'data','roi_names')
cd(currentdir);
disp('Job finished');
%end