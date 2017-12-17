close all, clear all, clc

%Baseline reward
baseReward = 965.64964;
% Data : steps, episode, total reward
data = load('rewardsDecisionFreq20MaxSteps1000');

th = title('Moving average of the total reward for a given episode');
xh = xlabel('Episodes');
yh = ylabel('Acumulated reward');
set([gca th xh yh],'fontsize',16,'fontweight','bold')
%plot(dataFreq20(:,2),dataFreq20(:,3))
dd = movmean(data(:,3),100); hold on
plot(data(:,2),dd,'linewidth',2)
plot(data(:,2),baseReward*ones(size(data(:,2))),'linewidth',2)

%%
close all, clear all, clc
% Data : steps, episode, total reward
data = load('rewardsDecisionFreq10MaxSteps1000');

th = title('Moving average of the total reward for a given episode');
xh = xlabel('Episodes');
yh = ylabel('Acumulated reward');
set([gca th xh yh],'fontsize',16,'fontweight','bold')
%plot(dataFreq20(:,2),dataFreq20(:,3))
dd = movmean(data(:,3),100); hold on
plot(data(:,2),dd,'linewidth',2)
