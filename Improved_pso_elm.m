
clc;
clear;
a=xlsread('traffic_900.csv');
% on line prediction vs direct transferability
% multi steps prediction
%*******************************************************
% on line version
%******************************************************
m1=max(a);
m2=min(a);
a=(a-min(a))/(max(a)-min(a));% normalization
No_of_Input=14; % input layer number of neurons
[n1,n2]=size(a);
data=zeros(n1-No_of_Input,No_of_Input+1);
for i=1:1:n1-No_of_Input
    data(i,:)=a(i:i+No_of_Input);
end

lo=0.05;
No_of_Output=2;
NumberofHiddenNeurons=20;
ActivationFunction='sigmoid';
popNum=50;
iteNum=200;
w=0.9;
c1=1;
c2=1;
fai=0.5;
conInterval=0.99;%%%%%%%%%%%***************************** change the confidence level*******
speed=2;
w1 = 11;%*******************
w2 = 0.1;
AA=xlsread('traffic_900.csv');

data=[data(:,No_of_Input+1)*(1-lo), data(:,No_of_Input+1)*(1+lo), data(:,1:No_of_Input)];
% data=[data(:,No_of_Input+1), data(:,1:No_of_Input)];
len1 = 600:15:885;
len2 = 615:15:900;

final_bestres = [];
final_bestInput = [];
final_bestHidden = [];
final_bestOutWei = [];
final_bestpre = [];
final_bestpsoIterRecord = [];

ol = 1;

for l=1:1:9
    % data=[quantity*(1-lo),quantity*(1+lo),day_item price];
    training=data(1:len1(l)-No_of_Input,:); % the training size may be changing based on the number of input layer
    testing=data(len1(l)+1-No_of_Input:len2(l)-No_of_Input,:); % the testing set is always the same

    bestscore = 10000;
    bestres = [];
    bestInput = [];
    bestHidden = [];
    bestOutWei = [];
    bestpre = [];
    bestpsoIterRecord = [];
    
    for i=1:1:100
        i
        [objVal_train,objVal_test, pre,flag,InputWeight,biasofHiddenNeurons,OutputWeight,A1,A,S,psoIterRecord,elmobjVal_test,elmflag,elmA1,elmA, elmS] = elm_pi( training,testing,No_of_Output,NumberofHiddenNeurons, ActivationFunction,popNum,iteNum,w,c1,c2,fai,conInterval,speed,lo,m1,m2, w1, w2, ol);
        
        if objVal_train<bestscore
            bestscore = objVal_train;
            bestres=[objVal_train,objVal_test,flag,A1,A,S,mean(pre(:,2)-pre(:,1)),elmobjVal_test,elmflag,elmA1,elmA,elmS];
            bestInput = InputWeight;
            bestHidden = biasofHiddenNeurons;
            bestOutWei = OutputWeight;
            bestpre = pre;
            bestpsoIterRecord = psoIterRecord;
        end
        bestres
    end
    l
    final_bestres = [final_bestres;bestres];
    final_bestInput = [final_bestInput;bestInput];
    final_bestHidden = [final_bestHidden;bestHidden];
    final_bestOutWei = [final_bestOutWei;bestOutWei];
    final_bestpre = [final_bestpre;bestpre];
    final_bestpsoIterRecord = [final_bestpsoIterRecord;bestpsoIterRecord];
end

xlswrite(strcat('olbestres_',num2str(No_of_Input),'_',num2str(NumberofHiddenNeurons),'_',num2str(No_of_Output),'_',num2str(conInterval),'_',num2str(iteNum),'_9','.xlsx'),final_bestres);
 xlswrite(strcat('olinputweight_test_',num2str(No_of_Input),'_',num2str(NumberofHiddenNeurons),'_',num2str(No_of_Output),'_',num2str(conInterval),'_',num2str(iteNum),'_9','.xlsx'),final_bestInput);
 xlswrite(strcat('olOutputWeight_test_',num2str(No_of_Input),'_',num2str(NumberofHiddenNeurons),'_',num2str(No_of_Output),'_',num2str(conInterval),'_',num2str(iteNum),'_9','.xlsx'),final_bestOutWei);
 xlswrite(strcat('olbiasofhidden_test_',num2str(No_of_Input),'_',num2str(NumberofHiddenNeurons),'_',num2str(No_of_Output),'_',num2str(conInterval),'_',num2str(iteNum),'_9','.xlsx'),final_bestHidden);
 xlswrite(strcat('olbestpre_',num2str(No_of_Input),'_',num2str(NumberofHiddenNeurons),'_',num2str(No_of_Output),'_',num2str(conInterval),'_',num2str(iteNum),'_9','.xlsx'),final_bestpre);
 xlswrite(strcat('olbestpsoIterRecord_',num2str(No_of_Input),'_',num2str(NumberofHiddenNeurons),'_',num2str(No_of_Output),'_',num2str(conInterval),'_',num2str(iteNum),'_9','.xlsx'),final_bestpsoIterRecord);
