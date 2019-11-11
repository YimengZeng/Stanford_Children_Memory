list=[15,12,9];
for i=1:24
    aa{i}=transition_subject_wise(estStatesCell{1,i},list);
end
rest2_transition=aa;
for i=1:24
    rest2(i,:)=reshape(rest2_transition{i}',1,9);
end
aaa=mean(aa);
reshape(aaa',3,3);
tran=[0.8454,0.1089,0.0458;0.1456,0.8087,0.0457;0.1505,0.1307,0.7188];
 
list=[15,12,9];
for i=1:24
    a{i}=transition_subject_wise(estStatesCell{i},list);
end
aaaa=compute_subject_level_transition_probabilities(model);
rest2_transition=a;
for i=1:24
   aa(i,:)=reshape(a{i}',1,9);
end
for j=1:9
    for i=1:5
    [corrE12_pr, pvalueE12_pr] = corr(ab(:,j), pdata(1:end,i), 'Type', 'Spearman');
    aba(j,i)=pvalueE12_pr;
    end
end

[corrE12_pr, pvalueE12_pr] = corr(ag(:,3),pdata(:,5), 'Type', 'Spearman') %Separate for Encoding 1 or 2
[corrE12_ml, pvalueE12_ml] = corr(mean_life(1:end,[15,12,9]), pdata(1:end,4),'Type','Spearman')
% [corrE12_cb, pvalueE12_cb] = corr(combined(1:end,1:5),E12_Acc(1:end,1),'Type','Spearman'); %[corrAll, pvalueAll] = corr(fractional_occupancy(:,1:5), pdata(:,1:5), 'Type', 'Pearson');
%x = E12_dprime(2:end,1); 
x = pdata(:,5);
y = ag(:,3); xfit = min(x):0.01:max(x); 
% y = mean_life(list,[12,15,9]);

figure(3); 
for ii = 1:1%size(y,2)
    num = strcat('23',num2str(ii));
    % subplot(num); 
    a = plot(x,y(:,ii),'o'); set(a,'MarkerEdgeColor','r','MarkerFaceColor','r');
    [p,s] = polyfit(x,y(:,ii),1); 
    [yfit,dy] = polyconf(p,xfit,s,'predopt','curve','simopt','on'); 
    line(xfit,yfit,'color','k','LineWidth',1)
    hold on
    plot(xfit,yfit-dy,'k:')
    plot(xfit,yfit+dy,'k:')
    p = []; s = [];
end
hit_t0=SME.fre_hit_state_t0;
miss_t0=SME.fre_miss_state_t0; 
hit_t01=SME.fre_hit_state_t01;
miss_t01=SME.fre_miss_state_t01; 
hit_t012=SME.fre_hit_state_t012;
miss_t012=SME.fre_miss_state_t012; 

for i=1:24
    a=sum(hit_t012(i,:));
    hit_t012(i,1)=hit_t012(i,1)/a;
    hit_t012(i,2)=hit_t012(i,2)/a;
    hit_t012(i,3)=hit_t012(i,3)/a;
end

for i=1:24
    a=sum(miss_t012(i,:));
    miss_t012(i,1)=miss_t012(i,1)/a;
    miss_t012(i,2)=miss_t012(i,2)/a;
    miss_t012(i,3)=miss_t012(i,3)/a;
end

hit=mean(hit_t012);
miss=mean(miss_t012);
hiterr=std(hit_t012)/sqrt(23);
misserr=std(miss_t012)/sqrt(23);
hm=[hit(1),miss(1);hit(2),miss(2);hit(3),miss(3)];
bar(hm)
hold on
errorbar(0.85,hit(1),hiterr(1), 'Color','k' )
errorbar(1.15,miss(1),misserr(1), 'Color','k' )
errorbar(1.85,hit(2),hiterr(2), 'Color','k' )
errorbar(2.15,miss(2),misserr(2), 'Color','k' )
errorbar(2.85,hit(3),hiterr(3), 'Color','k' )
errorbar(3.15,miss(3),misserr(3), 'Color','k' )
for i=1:24
    hit_t0(i,:)=hit_t0(i,:)/sum(hit_t0(i,:));
end

for i=1:24
    miss_t0(i,:)=miss_t0(i,:)/sum(miss_t0(i,:));
end

hit=mean(hit_t0);
miss=mean(miss_t012);
 hm=[hit(1),miss(1);hit(2),miss(2);hit(3),miss(3)];
 
 
 cccc=[32,56,100;49,178,163;255,255,0];
 cccc=[224,222,69;35,154,168;46,38,119];
 cccc=cccc/255;
 
figure(2)
MA = nanmean(fractional_occupancy)*100;
EA = nanstd(fractional_occupancy)*100;
MAA=MA([9,12,15]);
EAA=EA([9,12,15]);
ccc = distinguishable_colors(15);
ccc(15,:)=cccc(1,:);
ccc(12,:)=cccc(2,:);
ccc(9,:)=cccc(3,:);
for s=1:3
      bar(s,MAA(s),'FaceColor',cccc(s,:),'EdgeColor',cccc(s,:))
      hold on
      errorbar(s,MAA(s),EAA(s), 'Color',cccc(s,:) )
      hold on
end
title('occupancy rate')
MA2 = nanmean(mean_life);
EA2 = nanstd(mean_life);
MAA2=MA2([9,12,15]);
EAA2=EA2([9,12,15]);
subplot 212
for s=1:3
      bar(s,MAA2(s),'FaceColor',cccc(s,:),'EdgeColor',cccc(s,:))
      hold on
      errorbar(s,MAA2(s),EAA2(s), 'Color',cccc(s,:) )
      hold on
end
title('mean life')

r1=merged_occupancy_r1;
r2=merged_occupancy;
mr1=mean(r1);
mr2=mean(r2);
emr1(1)=std(r1(:,1))/sqrt(23);
emr1(2)=std(r1(:,2))/sqrt(23);
emr1(3)=std(r1(:,3))/sqrt(23);
emr2(1)=std(r2(:,1))/sqrt(23);
emr2(2)=std(r2(:,2))/sqrt(23);
emr2(3)=std(r2(:,3))/sqrt(23);
r2mr1=[mr1(1),mr2(1);mr1(2),mr2(2);mr1(3),mr2(3)];
bar(r2mr1);
hold on
errorbar(0.85,mr1(1),emr1(1), 'Color','k' );
errorbar(1.15,mr2(1),emr2(1), 'Color','k' );
errorbar(1.85,mr1(2),emr1(2), 'Color','k' );
errorbar(2.15,mr2(2),emr2(2), 'Color','k' );
errorbar(2.85,mr1(3),emr1(3), 'Color','k' );
errorbar(3.15,mr2(3),emr2(3), 'Color','k' );

figure(667);
for k=1:1
      ind = strcat('32',num2str(k));
      % subplot(1,3,k)
      % subplot(str2num(ind))
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
    cca_plotcausality(est_network,roi_names, 0.6);
    axis off
end

figure(1)
subplot(211);
imagesc(memory(2).onset_hit_miss');
caxis([-1,1]);
colorbar('EastOutside')
colormap=
subplot(212);
imagesc(estStatesCell{2});
colorbar('EastOutside')



figure(7);
subplot(311);
imagesc(estStatesCell{2});
pf2 = gca;
set(pf2,'YDir','normal')
cccc=[255,255,0;49,178,163;32,56,100];
cccc=cccc/255;
colormap(cccc)

a=strcat(num2str(hit_state_t0),num2str(hit_state_t1),num2str(hit_state_t2));
b=str2num(a);


for i=1:24
    for j=1:90
       hit_t012{i,j}=strcat(num2str(hit_state_t0(i,j)),num2str(hit_state_t1(i,j)),num2str(hit_state_t2(i,j)));
    end
end

for i=1:24
    for j=1:78
       miss_t012{i,j}=strcat(num2str(miss_state_t0(i,j)),num2str(miss_state_t1(i,j)),num2str(miss_state_t2(i,j)));
    end
end

for i=1:24
count=[0,0,0];
count0=0;
  for j=1:90
        if strcmp(hit_t012{i,j},'151515')
            count(1)=count(1)+1;
        end
        if strcmp(hit_t012{i,j},'121212')
            count(2)=count(2)+1;
        end
        if strcmp(hit_t012{i,j},'999')
            count(3)=count(3)+1;
        end
        if strcmp(hit_t012{i,j},'000')
            count0=count0+1;
        end
  end
  count1=90-sum(count)-count0;
  allcount_hit_t01(i,[1,2,3,4,5])=[count,count1,count0];
end

for i=1:24
count=[0,0,0];
count0=0;
  for j=1:78
        if strcmp(miss_t012{i,j},'151515')
            count(1)=count(1)+1;
        end
        if strcmp(miss_t012{i,j},'121212')
            count(2)=count(2)+1;
        end
        if strcmp(miss_t012{i,j},'999')
            count(3)=count(3)+1;
        end
        if strcmp(miss_t012{i,j},'000')
            count0=count0+1;
        end
  end
  count1=78-sum(count)-count0;
  allcount_miss_t01(i,[1,2,3,4,5])=[count,count1,count0];
end

for i=1:24
    allcount_hit(i,4)=allcount_hit(i,1)/(allcount_hit(i,1)+allcount_hit(i,2));
    allcount_hit(i,5)=allcount_hit(i,2)/(allcount_hit(i,1)+allcount_hit(i,2));
end

for i=1:24
    allcount_miss(i,4)=allcount_miss(i,1)/(allcount_miss(i,1)+allcount_miss(i,2));
    allcount_miss(i,5)=allcount_miss(i,2)/(allcount_miss(i,1)+allcount_miss(i,2));
end

static=[mean(allcount_hit(:,4)),mean(allcount_miss(:,4));mean(allcount_hit(:,5)),mean(allcount_miss(:,5))];
bar(static)
e1=std(allcount_hit(:,4))/sqrt(23);
e2=std(allcount_miss(:,4))/sqrt(23);
e3=std(allcount_hit(:,5))/sqrt(23);
e4=std(allcount_miss(:,5))/sqrt(23);
hold on
errorbar(0.85,mean(allcount_hit(:,4)),e1, 'Color','k' );
errorbar(1.15,mean(allcount_miss(:,4)),e2, 'Color','k' );
errorbar(1.85,mean(allcount_hit(:,5)),e3, 'Color','k' );
errorbar(2.15,mean(allcount_miss(:,5)),e4, 'Color','k' );

ct_t0=SME.fre_hit_state_t0;
ct_t0=ct_t0*436;
for i=1:24
    ct_t0(i,:)=ct_t0(i,:)/sum(ct_t0(i,:));
end

ct_t01=SME.fre_hit_state_t01;
ct_t01=ct_t01*436;
for i=1:24
    ct_t01(i,:)=ct_t01(i,:)/sum(ct_t01(i,:));
end

ct_t012=SME.fre_hit_state_t012;
ct_t012=ct_t012*436;
for i=1:24
    ct_t012(i,:)=ct_t012(i,:)/sum(ct_t012(i,:));
end

mt_t012=SME.fre_miss_state_t012;
mt_t012=mt_t012*436;
for i=1:24
    mt_t012(i,:)=mt_t012(i,:)/sum(mt_t012(i,:));
end

r1=allcount_hit_t01(:,[6,7,8]);
r2=allcount_miss_t01(:,[6,7,8]);
mr1=mean(r1);
mr2=mean(r2);
emr1(1)=std(r1(:,1))/sqrt(23);
emr1(2)=std(r1(:,2))/sqrt(23);
emr1(3)=std(r1(:,3))/sqrt(23);
emr2(1)=std(r2(:,1))/sqrt(23);
emr2(2)=std(r2(:,2))/sqrt(23);
emr2(3)=std(r2(:,3))/sqrt(23);
r2mr1=[mr1(1),mr2(1);mr1(2),mr2(2);mr1(3),mr2(3)];
bar(r2mr1);
hold on
errorbar(0.85,mr1(1),emr1(1), 'Color','k' );
errorbar(1.15,mr2(1),emr2(1), 'Color','k' );
errorbar(1.85,mr1(2),emr1(2), 'Color','k' );
errorbar(2.15,mr2(2),emr2(2), 'Color','k' );
errorbar(2.85,mr1(3),emr1(3), 'Color','k' );
errorbar(3.15,mr2(3),emr2(3), 'Color','k' );

for i=1:24
allcount_hit_t01(i,6)=allcount_hit_t01(i,1)/sum(allcount_hit_t01(i,[1,2,3]));
allcount_hit_t01(i,7)=allcount_hit_t01(i,2)/sum(allcount_hit_t01(i,[1,2,3]));
allcount_hit_t01(i,8)=allcount_hit_t01(i,3)/sum(allcount_hit_t01(i,[1,2,3]));
end

for i=1:24
allcount_miss_t01(i,6)=allcount_miss_t01(i,1)/sum(allcount_miss_t01(i,[1,2,3]));
allcount_miss_t01(i,7)=allcount_miss_t01(i,2)/sum(allcount_miss_t01(i,[1,2,3]));
allcount_miss_t01(i,8)=allcount_miss_t01(i,3)/sum(allcount_miss_t01(i,[1,2,3]));
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
    %  para=betweenness_wei(est_network);
       para1=strengths_und(input_cov_matx(:,:,1));
       para2=strengths_und(input_cov_matx(:,:,2));
       para3=strengths_und(input_cov_matx(:,:,3));
    %     BT = [BT bt' index_zero];
    %     ST = [ST st index_zero];
    %     axis square
    % imagesc(partial_corr(:,:,k) )
    %     caxis([0,0.3]);
    %     set(gca, 'YTick', [0 6, 7, 10,11, 14,15, 20], 'YTickLabel',[ ] );
    
      %  bar(para,'FaceColor',cccc(k,:),'EdgeColor','k')
      figure(1)
      bar(para2-para3)
        hold on
        %     bar(bt)
    set(gca, 'XTick',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],'XTickLabel',['R aHipp','L aHipp','R mHipp','L mHipp','R PHC','L PHC','R FG','L FG','R LOC','L LOC','R LPC','L LPC','R IFG','L IFG','PCC','MPFC']);
end


for i=1:24
    for j=1:90
       hit_t01{i,j}=strcat(num2str(hit_state_t0(i,j)),num2str(hit_state_t1(i,j)));
    end
end

for i=1:24
    for j=1:78
       miss_t01{i,j}=strcat(num2str(miss_state_t0(i,j)),num2str(miss_state_t1(i,j)));
    end
end

for i=1:24
count=[0,0,0,0,0,0,0,0];
count0=0;
  for j=1:90
        if strcmp(hit_t012{i,j},'151515')
            count(1)=count(1)+1;
        end
        if strcmp(hit_t012{i,j},'121212')
            count(2)=count(2)+1;
        end
        if strcmp(hit_t012{i,j},'151512')
            count(3)=count(3)+1;
        end
        if strcmp(hit_t012{i,j},'151212')
            count(4)=count(4)+1;
        end
        if strcmp(hit_t012{i,j},'121215')
            count(5)=count(5)+1;
        end
        if strcmp(hit_t012{i,j},'121515')
            count(6)=count(6)+1;
        end        
        if strcmp(hit_t012{i,j},'151215')
            count(7)=count(7)+1;
        end
        if strcmp(hit_t012{i,j},'121512')
            count(8)=count(8)+1;
        end
        if strcmp(hit_t012{i,j},'000')
            count0=count0+1;
        end
  end
  count1=90-sum(count)-count0;
  pathcount_hit_t012(i,:)=[count,count1,count0];
end

for i=1:24
count=[0,0,0,0,0,0,0,0];
count0=0;
  for j=1:78
         if strcmp(miss_t012{i,j},'151515')
            count(1)=count(1)+1;
        end
        if strcmp(miss_t012{i,j},'121212')
            count(2)=count(2)+1;
        end
        if strcmp(miss_t012{i,j},'151512')
            count(3)=count(3)+1;
        end
        if strcmp(miss_t012{i,j},'151212')
            count(4)=count(4)+1;
        end
        if strcmp(miss_t012{i,j},'121215')
            count(5)=count(5)+1;
        end
        if strcmp(miss_t012{i,j},'121515')
            count(6)=count(6)+1;
        end        
        if strcmp(miss_t012{i,j},'151215')
            count(7)=count(7)+1;
        end
        if strcmp(miss_t012{i,j},'121512')
            count(8)=count(8)+1;
        end
        if strcmp(miss_t012{i,j},'000')
            count0=count0+1;
        end
  end
  count1=78-sum(count)-count0;
  pathcount_miss_t012(i,:)=[count,count1,count0];
end

allhit=sum(sum(pathcount_hit_t012(:,[1:1:9]),2));
allmiss=sum(sum(pathcount_miss_t012(:,[1:1:9]),2));

pphit(1,1)=sum(pathcount_hit_t012(:,1))/allhit;
pphit(1,2)=sum(pathcount_hit_t012(:,2))/allhit;
pphit(1,3)=sum(pathcount_hit_t012(:,3))/allhit;
pphit(1,4)=sum(pathcount_hit_t012(:,4))/allhit;
pphit(1,5)=sum(pathcount_hit_t012(:,5))/allhit;
pphit(1,6)=sum(pathcount_hit_t012(:,6))/allhit;
pphit(1,7)=sum(pathcount_hit_t012(:,7))/allhit;
pphit(1,8)=sum(pathcount_hit_t012(:,8))/allhit;
pphit(1,9)=sum(pathcount_hit_t012(:,9))/allhit;

ppmiss(1,1)=sum(pathcount_miss_t012(:,1))/allmiss;
ppmiss(1,2)=sum(pathcount_miss_t012(:,2))/allmiss;
ppmiss(1,3)=sum(pathcount_miss_t012(:,3))/allmiss;
ppmiss(1,4)=sum(pathcount_miss_t012(:,4))/allmiss;
ppmiss(1,5)=sum(pathcount_miss_t012(:,5))/allmiss;
ppmiss(1,6)=sum(pathcount_miss_t012(:,6))/allmiss;
ppmiss(1,7)=sum(pathcount_miss_t012(:,7))/allmiss;
ppmiss(1,8)=sum(pathcount_miss_t012(:,8))/allmiss;
ppmiss(1,9)=sum(pathcount_miss_t012(:,9))/allmiss;

for i=1:24
allcount_hit_t01(i,6)=allcount_hit_t01(i,1)/sum(allcount_hit_t01(i,[1,2,3]));
allcount_hit_t01(i,7)=allcount_hit_t01(i,2)/sum(allcount_hit_t01(i,[1,2,3]));
allcount_hit_t01(i,8)=allcount_hit_t01(i,3)/sum(allcount_hit_t01(i,[1,2,3]));
end

for i=1:24
allcount_miss_t01(i,6)=allcount_miss_t01(i,1)/sum(allcount_miss_t01(i,[1,2,3]));
allcount_miss_t01(i,7)=allcount_miss_t01(i,2)/sum(allcount_miss_t01(i,[1,2,3]));
allcount_miss_t01(i,8)=allcount_miss_t01(i,3)/sum(allcount_miss_t01(i,[1,2,3]));
end


static=[mean(allcount_hit(:,4)),mean(allcount_miss(:,4));mean(allcount_hit(:,5)),mean(allcount_miss(:,5))];
bar(static)
e1=std(allcount_hit(:,4))/sqrt(23);
e2=std(allcount_miss(:,4))/sqrt(23);
e3=std(allcount_hit(:,5))/sqrt(23);
e4=std(allcount_miss(:,5))/sqrt(23);
hold on
errorbar(0.85,mean(allcount_hit(:,4)),e1, 'Color','k' );
errorbar(1.15,mean(allcount_miss(:,4)),e2, 'Color','k' );
errorbar(1.85,mean(allcount_hit(:,5)),e3, 'Color','k' );
errorbar(2.15,mean(allcount_miss(:,5)),e4, 'Color','k' );


for i=1:24
count=sum(abs(memory(i).onset_hit_miss(ord{i})));
s2(i,5)=sum(abs(memory(i).onset_hit_miss(ord{i})))-(sum(abs(memory(i).onset_hit_miss(ord{i})))-sum(memory(i).onset_hit_miss(ord{i})))/2;
s2(i,5)=s2(i,5)/count;
s2(i,6)=(sum(abs(memory(i).onset_hit_miss(ord{i})))-sum(memory(i).onset_hit_miss(ord{i})))/2;
s2(i,6)=s2(i,6)/count;
end




% adding t minus 1 data
allcount=zeros(1,24);
for j=1:24
    count=0;
 for i=1:90
    if hit_state_t0(j,i)~=0
        count=count+1;
    end
 end
 allcount(j)=count;
end

for i=1:24
    z=[hit_state_tm1(i,:),hit_state_t0(i,:),hit_state_t1(i,:)];
    ahit(i,1)=length(find(z(:)==15))/(3*allcount(i));
    ahit(i,2)=length(find(z(:)==12))/(3*allcount(i));
    ahit(i,3)=length(find(z(:)==9))/(3*allcount(i));
end


allcount=zeros(1,24);
for j=1:24
    count=0;
 for i=1:78
    if miss_state_t0(j,i)~=0
        count=count+1;
    end
 end
 allcount(j)=count;
end

for i=1:24
    z1=[miss_state_tm1(i,:),miss_state_t0(i,:),miss_state_t1(i,:)];
    amiss(i,1)=length(find(z1(:)==15))/(3*allcount(i));
    amiss(i,2)=length(find(z1(:)==12))/(3*allcount(i));
    amiss(i,3)=length(find(z1(:)==9))/(3*allcount(i));
end

for i=1:24
    for j=1:90
       hit_tm0{i,j}=strcat(num2str(hit_state_t0(i,j)),num2str(hit_state_t1(i,j)));
    end
end

for i=1:24
count=[0,0,0];
count0=0;
  for j=1:90
        if strcmp(hit_tm0{i,j},'1515')
            count(1)=count(1)+1;
        end
        if strcmp(hit_tm0{i,j},'1212')
            count(2)=count(2)+1;
        end
        if strcmp(hit_tm0{i,j},'99')
            count(3)=count(3)+1;
        end
        if strcmp(hit_tm0{i,j},'00')
            count0=count0+1;
        end
  end
  count1=90-sum(count)-count0;
  allcount_hit_tm0(i,:)=[count,count1,count0];
end

for i=1:24
    zz1(i,1)=allcount_hit_tm0(i,1)/sum(allcount_hit_tm0(i,[1,2,3,4]));
    zz1(i,2)=allcount_hit_tm0(i,2)/sum(allcount_hit_tm0(i,[1,2,3,4]));
    zz1(i,3)=allcount_hit_tm0(i,3)/sum(allcount_hit_tm0(i,[1,2,3,4]));
    zz1(i,4)=allcount_hit_tm0(i,4)/sum(allcount_hit_tm0(i,[1,2,3,4]));
end



for i=1:24
    for j=1:78
       miss_tm0{i,j}=strcat(num2str(miss_state_t0(i,j)),num2str(miss_state_t1(i,j)));
    end
end

for i=1:24
count=[0,0,0];
count0=0;
  for j=1:78
        if strcmp(miss_tm0{i,j},'1515')
            count(1)=count(1)+1;
        end
        if strcmp(miss_tm0{i,j},'1212')
            count(2)=count(2)+1;
        end
        if strcmp(miss_tm0{i,j},'99')
            count(3)=count(3)+1;
        end
        if strcmp(miss_tm0{i,j},'00')
            count0=count0+1;
        end
  end
  count1=78-sum(count)-count0;
  allcount_miss_tm0(i,:)=[count,count1,count0];
end



for i=1:24
    zz2(i,1)=allcount_miss_tm0(i,1)/sum(allcount_miss_tm0(i,[1,2,3,4]));
    zz2(i,2)=allcount_miss_tm0(i,2)/sum(allcount_miss_tm0(i,[1,2,3,4]));
    zz2(i,3)=allcount_miss_tm0(i,3)/sum(allcount_miss_tm0(i,[1,2,3,4]));
    zz2(i,4)=allcount_miss_tm0(i,4)/sum(allcount_miss_tm0(i,[1,2,3,4]));
end

% sliding window is 2
for i=1:24
    for j=1:90
       hit_tm0{i,j}=strcat(num2str(hit_state_t1(i,j)),num2str(hit_state_t2(i,j)));
    end
end

for i=1:24
count=zeros(1,9);
count0=0;
  for j=1:90
        if strcmp(hit_tm0{i,j},'1515')
            count(1)=count(1)+1;
        end
        if strcmp(hit_tm0{i,j},'1212')
            count(2)=count(2)+1;
        end
        if strcmp(hit_tm0{i,j},'99')
            count(3)=count(3)+1;
        end
        if strcmp(hit_tm0{i,j},'1512')
            count(4)=count(4)+1;
        end
        if strcmp(hit_tm0{i,j},'1215')
            count(5)=count(5)+1;
        end
        if strcmp(hit_tm0{i,j},'159')
            count(6)=count(6)+1;
        end
        if strcmp(hit_tm0{i,j},'915')
            count(7)=count(7)+1;
        end
        if strcmp(hit_tm0{i,j},'129')
            count(8)=count(8)+1;
        end
        if strcmp(hit_tm0{i,j},'912')
            count(9)=count(9)+1;
        end
        if strcmp(hit_tm0{i,j},'00')
            count0=count0+1;
        end
  end
  count1=90-sum(count)-count0;
  allcount_hit_tm0(i,:)=[count,count1,count0];
end

for i=1:24
    aa1(i,1)=allcount_hit_tm0(i,1)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,2)=allcount_hit_tm0(i,2)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,3)=allcount_hit_tm0(i,3)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,4)=allcount_hit_tm0(i,4)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,5)=allcount_hit_tm0(i,5)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,6)=allcount_hit_tm0(i,6)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,7)=allcount_hit_tm0(i,7)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,8)=allcount_hit_tm0(i,8)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,9)=allcount_hit_tm0(i,9)/sum(allcount_hit_tm0(i,1:end-1));
end



for i=1:24
    for j=1:78
       miss_tm0{i,j}=strcat(num2str(miss_state_t1(i,j)),num2str(miss_state_t2(i,j)));
    end
end

for i=1:24
count=zeros(1,9);
count0=0;
  for j=1:78
        if strcmp(miss_tm0{i,j},'1515')
            count(1)=count(1)+1;
        end
        if strcmp(miss_tm0{i,j},'1212')
            count(2)=count(2)+1;
        end
        if strcmp(miss_tm0{i,j},'99')
            count(3)=count(3)+1;
        end
        if strcmp(miss_tm0{i,j},'1512')
            count(4)=count(4)+1;
        end
        if strcmp(miss_tm0{i,j},'1215')
            count(5)=count(5)+1;
        end
        if strcmp(miss_tm0{i,j},'159')
            count(6)=count(6)+1;
        end
        if strcmp(miss_tm0{i,j},'915')
            count(7)=count(7)+1;
        end
        if strcmp(miss_tm0{i,j},'129')
            count(8)=count(8)+1;
        end
        if strcmp(miss_tm0{i,j},'912')
            count(9)=count(9)+1;
        end
        if strcmp(miss_tm0{i,j},'00')
            count0=count0+1;
        end
  end
  count1=78-sum(count)-count0;
  allcount_miss_tm0(i,:)=[count,count1,count0];
end



for i=1:24
    aa2(i,1)=allcount_miss_tm0(i,1)/sum(allcount_miss_tm0(i,1:end-1));
    aa2(i,2)=allcount_miss_tm0(i,2)/sum(allcount_miss_tm0(i,1:end-1));
    aa2(i,3)=allcount_miss_tm0(i,3)/sum(allcount_miss_tm0(i,1:end-1));
    aa2(i,4)=allcount_miss_tm0(i,4)/sum(allcount_miss_tm0(i,1:end-1));
    aa2(i,5)=allcount_miss_tm0(i,5)/sum(allcount_miss_tm0(i,1:end-1));
    aa2(i,6)=allcount_miss_tm0(i,6)/sum(allcount_miss_tm0(i,1:end-1));
    aa2(i,7)=allcount_miss_tm0(i,7)/sum(allcount_miss_tm0(i,1:end-1));
    aa2(i,8)=allcount_miss_tm0(i,8)/sum(allcount_miss_tm0(i,1:end-1));  
    aa2(i,9)=allcount_miss_tm0(i,9)/sum(allcount_miss_tm0(i,1:end-1));
end

aa11=mean(aa1);
aa22=mean(aa2);
bar([aa11(1),aa22(1);aa11(2),aa22(2);aa11(3),aa22(3);aa11(4),aa22(4);aa11(5),aa22(5);aa11(6),aa22(6);aa11(7),aa22(7);aa11(8),aa22(8);aa11(9),aa22(9)])

% sliding window is 3

for i=1:24
    for j=1:90
       hit_tm0{i,j}=strcat(num2str(hit_state_t1(i,j)),num2str(hit_state_t2(i,j)),num2str(hit_state_t3(i,j)));
    end
end

for i=1:24
count=zeros(1,9);
count0=0;
  for j=1:90
        if strcmp(hit_tm0{i,j},'151515')
            count(1)=count(1)+1;
        end
        if strcmp(hit_tm0{i,j},'121212')
            count(2)=count(2)+1;
        end
        if strcmp(hit_tm0{i,j},'151512')
            count(3)=count(3)+1;
        end
        if strcmp(hit_tm0{i,j},'151212')
            count(4)=count(4)+1;
        end
        if strcmp(hit_tm0{i,j},'121515')
            count(5)=count(5)+1;
        end
        if strcmp(hit_tm0{i,j},'121512')
            count(6)=count(6)+1;
        end
        if strcmp(hit_tm0{i,j},'151215')
            count(7)=count(7)+1;
        end
        if strcmp(hit_tm0{i,j},'15129')
            count(8)=count(8)+1;
        end
        if strcmp(hit_tm0{i,j},'12159')
            count(9)=count(9)+1;
        end
        if strcmp(hit_tm0{i,j},'000')
            count0=count0+1;
        end
  end
  count1=90-sum(count)-count0;
  allcount_hit_tm0(i,:)=[count,count1,count0];
end

for i=1:24
    aa1(i,1)=allcount_hit_tm0(i,1)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,2)=allcount_hit_tm0(i,2)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,3)=allcount_hit_tm0(i,3)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,4)=allcount_hit_tm0(i,4)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,5)=allcount_hit_tm0(i,5)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,6)=allcount_hit_tm0(i,6)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,7)=allcount_hit_tm0(i,7)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,8)=allcount_hit_tm0(i,8)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,9)=allcount_hit_tm0(i,9)/sum(allcount_hit_tm0(i,1:end-1));
    aa1(i,10)=allcount_hit_tm0(i,10)/sum(allcount_hit_tm0(i,1:end-1));
end



for i=1:24
    for j=1:78
       miss_tm0{i,j}=strcat(num2str(miss_state_t1(i,j)),num2str(miss_state_t2(i,j)),num2str(miss_state_t3(i,j)));
    end
end

for i=1:24
count=zeros(1,9);
count0=0;
  for j=1:78
        if strcmp(miss_tm0{i,j},'151515')
            count(1)=count(1)+1;
        end
        if strcmp(miss_tm0{i,j},'121212')
            count(2)=count(2)+1;
        end
        if strcmp(miss_tm0{i,j},'151512')
            count(3)=count(3)+1;
        end
        if strcmp(miss_tm0{i,j},'151212')
            count(4)=count(4)+1;
        end
        if strcmp(miss_tm0{i,j},'121515')
            count(5)=count(5)+1;
        end
        if strcmp(miss_tm0{i,j},'121512')
            count(6)=count(6)+1;
        end
        if strcmp(miss_tm0{i,j},'151215')
            count(7)=count(7)+1;
        end
        if strcmp(miss_tm0{i,j},'15129')
            count(8)=count(8)+1;
        end
        if strcmp(miss_tm0{i,j},'12159')
            count(9)=count(9)+1;
        end
        if strcmp(miss_tm0{i,j},'000')
            count0=count0+1;
        end
  end
  count1=78-sum(count)-count0;
  allcount_miss_tm0(i,:)=[count,count1,count0];
end


for i=1:24
    aa2(i,1)=allcount_miss_tm0(i,1)/sum(allcount_miss_tm0(i,1:end-1));
    aa2(i,2)=allcount_miss_tm0(i,2)/sum(allcount_miss_tm0(i,1:end-1));
    aa2(i,3)=allcount_miss_tm0(i,3)/sum(allcount_miss_tm0(i,1:end-1));
    aa2(i,4)=allcount_miss_tm0(i,4)/sum(allcount_miss_tm0(i,1:end-1));
    aa2(i,5)=allcount_miss_tm0(i,5)/sum(allcount_miss_tm0(i,1:end-1));
    aa2(i,6)=allcount_miss_tm0(i,6)/sum(allcount_miss_tm0(i,1:end-1));
    aa2(i,7)=allcount_miss_tm0(i,7)/sum(allcount_miss_tm0(i,1:end-1));
    aa2(i,8)=allcount_miss_tm0(i,8)/sum(allcount_miss_tm0(i,1:end-1));  
    aa2(i,9)=allcount_miss_tm0(i,9)/sum(allcount_miss_tm0(i,1:end-1));
    aa2(i,10)=allcount_miss_tm0(i,10)/sum(allcount_miss_tm0(i,1:end-1));
end

aa11=mean(aa1);
aa22=mean(aa2);
bar([aa11(1),aa22(1);aa11(2),aa22(2);aa11(3),aa22(3);aa11(4),aa22(4);aa11(5),aa22(5);aa11(6),aa22(6);aa11(7),aa22(7);aa11(8),aa22(8);aa11(9),aa22(9);aa11(10),aa22(10)])

for i=1:24
ord1=ord{i}(:);
count=sum(abs(memory(i).onset_hit_miss(ord1)));
s2(i,3)=sum(abs(memory(i).onset_hit_miss(ord1)));
s2(i,3)=s2(i,3)/length(ord1);
end

% compute mean-life time based hit percentage
for i=1:24
    ord{i}=zeros(1,1);
    n=1;
    for j=1:433
        if estStatesCell{i}(j)==15 & estStatesCell{i}(j+1)==15 & estStatesCell{i}(j+2)==15 & estStatesCell{i}(j+3)==15
            k=j;
            while k~=436 & estStatesCell{i}(k)==15 & isempty(find(ord{i}(:)==k))        
                ord{i}(n)=k;
                n=n+1;
                k=k+1;
            end
        end
    end
end

for i=1:24
    ord{i}=zeros(1,1);
    n=1;
    for j=1:434
        if estStatesCell{i}(j)==12 & estStatesCell{i}(j+1)==12 & estStatesCell{i}(j+2)==12
            k=j;
            while k~=436 & estStatesCell{i}(k)==12 & isempty(find(ord{i}(:)==k))        
                ord{i}(n)=k;
                n=n+1;
                k=k+1;
            end
        end
    end
end

for i=1:24
    ord{i}=zeros(1,1);
    n=1;
    for j=1:435
        if estStatesCell{i}(j)==9 & estStatesCell{i}(j+1)==9
            k=j;
            while k~=436 & estStatesCell{i}(k)==9 & isempty(find(ord{i}(:)==k))        
                ord{i}(n)=k;
                n=n+1;
                k=k+1;
            end
        end
    end
end

for i=1:24
ord{i}=find(estStatesCell{i}(:)==4);
end

for i=1:24
count=sum(abs(memory(i).onset_hit_miss(ord{i})));
a=sum(abs(memory(i).onset_hit_miss(ord{i})))-(sum(abs(memory(i).onset_hit_miss(ord{i})))-sum(memory(i).onset_hit_miss(ord{i})))/2;
b=(sum(abs(memory(i).onset_hit_miss(ord{i})))-sum(memory(i).onset_hit_miss(ord{i})))/2;
s2(i,5)=a/(a+b);
end


% compute confidence related information

for i=1:24
ord1=find(estStatesCell{i}(:)==3);
confi1{i}(1)=length(find(memory(i).confidence_bi(ord1)==4));
confi1{i}(2)=length(find(memory(i).confidence_bi(ord1)==3));
confi1{i}(3)=length(find(memory(i).confidence_bi(ord1)==2));
confi1{i}(4)=length(find(memory(i).confidence_bi(ord1)==1));

end

for i=1:24
ord1=find(estStatesCell{i}(:)==9);
confi2{i}(1)=length(find(memory(i).confidence_bi(ord1)==4));
confi2{i}(2)=length(find(memory(i).confidence_bi(ord1)==3));
confi2{i}(3)=length(find(memory(i).confidence_bi(ord1)==2));
confi2{i}(4)=length(find(memory(i).confidence_bi(ord1)==1));

end

for i=1:24
ord1=find(estStatesCell{i}(:)==15);
confi3{i}(1)=length(find(memory(i).confidence_bi(ord1)==4));
confi3{i}(2)=length(find(memory(i).confidence_bi(ord1)==3));
confi3{i}(3)=length(find(memory(i).confidence_bi(ord1)==2));
confi3{i}(4)=length(find(memory(i).confidence_bi(ord1)==1));

end

for i=1:24
ord1=find(estStatesCell{i}(:)==6);
confi4{i}(1)=length(find(memory(i).confidence_bi(ord1)==4));
confi4{i}(2)=length(find(memory(i).confidence_bi(ord1)==3));
confi4{i}(3)=length(find(memory(i).confidence_bi(ord1)==2));
confi4{i}(4)=length(find(memory(i).confidence_bi(ord1)==1));

end

bar([aaa(:,1)';aaa(:,2)';aaa(:,3)';aaa(:,4)'])

for i=1:24
    cconfi4(i,:)=confi4{i};
end

for i=1:24
    cconfi4(i,:)=cconfi4(i,:)./sum(cconfi4(i,:));
end

aaa(1,:)=nanmean(cconfi1);
aaa(2,:)=nanmean(cconfi2);
aaa(3,:)=nanmean(cconfi3);
aaa(4,:)=nanmean(cconfi4);

% compute confidence related information
for i=1:24
ord1=find(estStatesCell{i}(:)==15);
confi1{i}(1)=length(find(memory(i).confidence_bi(ord1)==4));
confi1{i}(2)=length(find(memory(i).confidence_bi(ord1)==3));
confi1{i}(3)=length(find(memory(i).confidence_bi(ord1)==2));
confi1{i}(4)=length(find(memory(i).confidence_bi(ord1)==1));
end

for i=1:24
ord1=find(estStatesCell{i}(:)==12);
confi2{i}(1)=length(find(memory(i).confidence_bi(ord1)==4));
confi2{i}(2)=length(find(memory(i).confidence_bi(ord1)==3));
confi2{i}(3)=length(find(memory(i).confidence_bi(ord1)==2));
confi2{i}(4)=length(find(memory(i).confidence_bi(ord1)==1));
end

for i=1:24
ord1=find(estStatesCell{i}(:)==9);
confi3{i}(1)=length(find(memory(i).confidence_bi(ord1)==4));
confi3{i}(2)=length(find(memory(i).confidence_bi(ord1)==3));
confi3{i}(3)=length(find(memory(i).confidence_bi(ord1)==2));
confi3{i}(4)=length(find(memory(i).confidence_bi(ord1)==1));
end

for i=1:24
    cconfi3(i,:)=confi3{i};
end

for i=1:24
    cconfi3(i,:)=cconfi3(i,:)./sum(cconfi3(i,:));
end

aaa(1,:)=mean(ccconfi1(:,[1,5]));
aaa(2,:)=mean(ccconfi2(:,[1,5]));
aaa(3,:)=mean(ccconfi3(:,[1,5]));

for i=1:24
    ccconfi3(i,5)=ccconfi3(i,2)+ccconfi3(i,3)+ccconfi3(i,4);
end

%based on trial to fine confidence highest\high

for i=1:24
   ord=find(memory(i).confidence_bi(:)==4);
   a(i,[1,2,3])=sum(model.posterior_probabilities{i}(ord,[15,12,9]));
   clear 'ord'
   ord=find(memory(i).confidence_bi(:)==3);
   a(i,[4,5,6])=sum(model.posterior_probabilities{i}(ord,[15,12,9]));
   clear 'ord'
   ord=find(memory(i).confidence_bi(:)==2);
   a(i,[7,8,9])=sum(model.posterior_probabilities{i}(ord,[15,12,9]));
   clear 'ord'
   ord=find(memory(i).confidence_bi(:)==1);
   a(i,[10,11,12])=sum(model.posterior_probabilities{i}(ord,[15,12,9]));
   clear 'ord'
   ord=find(memory(i).confidence_bi(:)==-1);
   a(i,[13,14,15])=sum(model.posterior_probabilities{i}(ord,[15,12,9]));
   clear 'ord'
   ord=find(memory(i).confidence_bi(:)==-2);
   a(i,[16,17,18])=sum(model.posterior_probabilities{i}(ord,[15,12,9]));
   clear 'ord'
   ord=find(memory(i).confidence_bi(:)==-3);
   a(i,[19,20,21])=sum(model.posterior_probabilities{i}(ord,[15,12,9]));
   clear 'ord'
   ord=find(memory(i).confidence_bi(:)==-4);
   a(i,[22,23,24])=sum(model.posterior_probabilities{i}(ord,[15,12,9]));
   clear 'ord'  
   a(i,[1,4,7,10,13,16,19,22])=a(i,[1,4,7,10,13,16,19,22])./sum(a(i,[1,4,7,10,13,16,19,22]));
   a(i,[2,5,8,11,14,17,20,23])=a(i,[2,5,8,11,14,17,20,23])./sum(a(i,[2,5,8,11,14,17,20,23]));
   a(i,[3,6,9,12,15,18,21,24])=a(i,[3,6,9,12,15,18,21,24])./sum(a(i,[3,6,9,12,15,18,21,24]));
end
% network analysis

for i=1:24
bet{3}(i,:)=betweenness_bin(subj_cov_matx{1,i}(:,:,3));
end

aa(:,1)=bet{1};
aa(:,2)=bet{2};
aa(:,3)=bet{3};

MA = nanmean(aa);
EA = nanstd(aa);
EA=EA/sqrt(24);
ccc = distinguishable_colors(15);
sc=ccc([15,12,9],:);
for s=1:3
      bar(s,MA(s),'FaceColor',sc(s,:),'EdgeColor',sc(s,:))
      hold on
      errorbar(s,MA(s),EA(s), 'Color',sc(s,:) )
      hold on
end
title('efficiency')

for i=1:24
bet{3}(i,:)=efficiency_bin(subj_cov_matx{1,i}(:,:,3));
end


for i=1:16
[h,p]=ttest(bet{1,2}(:,i),bet{1,3}(:,i))
c(i)=p;
end

clustering_coef_bu(subj_cov_matx{1,1}(:,:,3))

for i=1:24
[Ci,Q]=modularity_und(subj_cov_matx{1,i}(:,:,1),1);
cet{1}(i,:)=participation_coef(subj_cov_matx{1,i}(:,:,1),Ci,0);
clear 'Ci' 'Q'
end

cca_plotcausality(Ci,roi_names2,.4)

