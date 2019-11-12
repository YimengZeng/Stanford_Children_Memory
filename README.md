# Stanford_Children_Memory_Project
Lead author(s): szqin
Created by zeng yi meng on November, 2019



A.	Behavioral data

Behavioral data (outside scanner)
o	Used ID sequence is according to ‘subjectslist_n24.txt’
o	In Github called ‘subjectlist_performance_n24.csv’ and ‘subjectlist_performance_n24.txt’ (txt including subject ID and their memory scores but the sequence of subject ID and their behavior data is not the same with fmri data (.csv has the right sequence), just make sure fmri time series、behavior data has correct paired sequence before analysis. Also, we provide behavior data in m format on Github called ‘subject_behavior_data.mat’.
o	Or can be found in /oak/stanford/groups/menon/projects/szqin/2015_memory_consolidation/dynamic_states/scripts_BSDS_sherlock/…/Group analysis/Zeng



Behavioral data from fMRI tasks
o	In Github called ‘onset_hit_miss_confidence_EncodingE1E2_n24_new.mat’ including Correct and Wrong and Confidence Rate for each trial.


B.	fMRI data and BSDS anslysis

•	brain activation map and task design 
o	Can be found in /oak/stanford/groups/menon/projects/szqin/2015_memory_consolidation/dynamic_states/scripts_BSDS_sherlock/
o	Coordination of 16 roi can be found in ‘roi_coordinations.txt’

•	fMRI data of all subjects and their ID
 
o	Time series of 16 roi for each subject is packaged into one m file on Github called ‘16ROI_ts_0a25encoding12_DMN_n24_latest.mat’
o	All subjects fmri raw data is storaged in regular path, just use subject ID to find.
o	ID of subjects is listed in ‘subjectslist_n24.txt’


•	BSDS training procedure
o	Step 1: extracting time series based on roi list using ‘
‘extract_ROI_TS_group_wm_csf_mnvt_swaor_nofl_enc12_218scans.m’
o	Step 2: training BSDS model, using ‘main_BSDS_singleinit.m’ and ‘main_BSDS_afterinit.m’

•	Figures plot and computations in code
o	Major code is in ‘group_analysis_community_detection_JT_qin_tmp_E12_16ROIs.m’
o	Other part of codes used in this article are in  ‘Stanford_project_supplementary_codes.m’
o	Note that in this article, when calculating transition probabilities, only heuristic approach shows significant results (Jalia’s formula didn’t show significant results). Related function is in Github called ‘transition_subject_wise.m’

C.	Post-encoding replay analysis related to post-encoding rest 
Trained BSDS Model employed here as decoder :
o	In Github called ‘model_EncodingE1E2_regMov_mean_dt1_24ss_raw_16ROIs_ss1to24_latest.mat’
o	Raw data is in the same path with encoding data mentioned before in Sherlock servers.
o	Codes of analysis of resting data is in ‘individual_log_likelihood_maximum_states_E12Rest.m’ This m file include  


 
D.	Supplementary analysis
o	All codes employed are in ‘group_analysis_community_detection_JT_qin_tmp_E12_16ROIs.m’ and ‘Stanford_project_supplementary_codes.m’


