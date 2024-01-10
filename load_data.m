clear;clc;close all;
a = load('final_data2.mat');
x = [];
y_desired = [];
for i = 1:574
    x(i,1) = a.team_id(i);
    x(i,2) = a.team_id(i);
    x(i,3) = a.opp_team_id(i);
    x(i,4) = a.toss_decision(i);
    y_desired(i) = a.match_winner_id(i);    
end
y_desired = reshape(y_desired,[574,1]);
y_desired = categorical(y_desired);


[m,n] = size(x);
P_train = 0.80;
P_test = 0.05;
idx = randperm(m);
X_Training = x(idx(1:round(P_train*m)),:) ; 
X_Test = x(idx(round(P_train*m)+1:round((P_train + P_test)*m)),:) ; 
X_val = x(idx(round((P_train + P_test)*m)+1:end),:) ;
labels_Training = y_desired(idx(1:round(P_train*m))) ; 
labels_Test = y_desired(idx(round(P_train*m)+1:round((P_train + P_test)*m))) ; 
labels_val = y_desired(idx(round((P_train + P_test)*m)+1:end)) ;
