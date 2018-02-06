clc;
clear;
a=xlsread('traffic_900.csv');

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
data=[data(:,No_of_Input+1)*(1-lo), data(:,No_of_Input+1)*(1+lo), data(:,1:No_of_Input)];

training=data(1:600-No_of_Input,:); % the training size may be changing based on the number of input layer
testing=data(601-No_of_Input:900-No_of_Input,:); % the testing set is always the same
No_of_Output=2;
NumberofHiddenNeurons=20;
ActivationFunction='sigmoid';
popNum=50;
iteNum=200;
%    1) 0 < (C1 + C2) < 4
%    2) (C1 + C2)/2 - 1 < w < 1
w=0.9;
c1=1;
c2=1;
fai=0.5;
conInterval=0.95;% change the confidence level
speed=2;
res=[];
w1=11;
w2=0.02;

AA=xlsread('traffic_900.csv');

bestscore = 10000;
bestres = [];
bestInput = [];
bestHidden = [];
bestOutWei = [];
bestpre = [];
bestpsoIterRecord = [];

for i=1:1:100
    i
[objVal_train,objVal_test, pre,flag,InputWeight,biasofHiddenNeurons,OutputWeight,A1,A,S,psoIterRecord,elmobjVal_test,elmflag,elmA1,elmA, elmS] = elm_pi( training,testing,No_of_Output,NumberofHiddenNeurons, ActivationFunction,popNum,iteNum,w,c1,c2,fai,conInterval,speed,lo,m1,m2, w1, w2);

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
res=[res;[objVal_train,objVal_test,flag,A1,A,S,mean(pre(:,2)-pre(:,1))]];

end

 xlswrite(strcat('bestres_',num2str(No_of_Input),'_',num2str(NumberofHiddenNeurons),'_',num2str(No_of_Output),'_',num2str(conInterval),'_',num2str(iteNum),'_07192017','.xlsx'),bestres);
 xlswrite(strcat('inputweight_test_',num2str(No_of_Input),'_',num2str(NumberofHiddenNeurons),'_',num2str(No_of_Output),'_',num2str(conInterval),'_',num2str(iteNum),'_07192017','.xlsx'),bestInput);
 xlswrite(strcat('OutputWeight_test_',num2str(No_of_Input),'_',num2str(NumberofHiddenNeurons),'_',num2str(No_of_Output),'_',num2str(conInterval),'_',num2str(iteNum),'_07192017','.xlsx'),bestOutWei);
 xlswrite(strcat('biasofhidden_test_',num2str(No_of_Input),'_',num2str(NumberofHiddenNeurons),'_',num2str(No_of_Output),'_',num2str(conInterval),'_',num2str(iteNum),'_07192017','.xlsx'),bestHidden);
 xlswrite(strcat('bestpre_',num2str(No_of_Input),'_',num2str(NumberofHiddenNeurons),'_',num2str(No_of_Output),'_',num2str(conInterval),'_',num2str(iteNum),'_07192017','.xlsx'),bestpre);
 xlswrite(strcat('bestpsoIterRecord_',num2str(No_of_Input),'_',num2str(NumberofHiddenNeurons),'_',num2str(No_of_Output),'_',num2str(conInterval),'_',num2str(iteNum),'_07192017','.xlsx'),bestpsoIterRecord);
