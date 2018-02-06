function [ globOptimal,bestValue, psoIterRecord ] = pso( ini_outputWeight,resHidden,popNum,iteNum,Target,conInterval,speed,w,c1,c2,fai,lo,m1,m2, w1, w2,ol)
% ini_outputWeight in Extreme Learning machine, the initial result
% resHidden, result from hidden layer, or the input of the output layer
% popNum, population number in PSO algorithm
% iteNum, iteration number
% Target, the real values
% conInterval, confidence interval
% speed, the maximum of speed

% resHidden num_hidden_neurons by total_observations
% ini_outputWeight num_hidden_neurons by num_output_neurons
% target number_output_neurons by total_observations



[sizeHidden,sizeOutput]=size(ini_outputWeight);
% generate popNumber random outputWeights
outputWeightPop=zeros(sizeHidden,sizeOutput,popNum);
for i=1:1:popNum
    outputWeightPop(:,:,i)=(rand(sizeHidden,sizeOutput)*1-0.5); % [-10,10]
    outputWeightPop(:,:,i)=outputWeightPop(:,:,i)+ini_outputWeight;
end

localOptimal=outputWeightPop;% record the best location for each population
globOptimal=zeros(sizeHidden,sizeOutput);% record the global best location

vPop=zeros(sizeHidden,sizeOutput,popNum); % speed 
psoIterRecord=[];% record the best PSO object value for each iteration
% initialization
for i=1:1:popNum
    vPop(:,:,i)=rand(sizeHidden,sizeOutput)*2*speed-speed; % [-speed,speed]
end

objectValue=ones(popNum,1)*100000; % record the object values for each population, initialized as a big number
reliValue=ones(popNum,1)*100000;
normsharpV=ones(popNum,1)*100000;
% unnormsharpV=ones(popNum,1)*100000;
for j=1:1:iteNum
    for i=1:1:popNum
        preInterval=(resHidden'*outputWeightPop(:,:,i))'; % the output of ELM
        [ objVal,flag, reliability, unabsrelia, normsharp]= elm_calObject( preInterval,Target(1,:)/(1-lo), conInterval,m1,m2, w1, w2, ol);
        if flag==0 && objectValue(i)>objVal % flag is 1 means the interval is wrong
            objectValue(i)=objVal; %update the minimumm object value;
            reliValue(i)=reliability;
            normsharpV(i)=normsharp;
%             unnormsharpV(i)=sharpness;
            localOptimal(:,:,i)=outputWeightPop(:,:,i);
        end
    end
    [bestValue,bestIndex]=min(objectValue);
    psoIterRecord=[psoIterRecord;[bestValue,reliValue(bestIndex),normsharpV(bestIndex)]];
    
    globOptimal=localOptimal(:,:,bestIndex);
    
    % update the locations of the particles
    for i=1:1:popNum
        vPop(:,:,i)=w*vPop(:,:,i)+c1*rand(1)*(localOptimal(:,:,i)-outputWeightPop(:,:,i))+c2*rand(1)*(globOptimal-outputWeightPop(:,:,i));
        outputWeightPop(:,:,i)=outputWeightPop(:,:,i)+fai*vPop(:,:,i);
    end
end
end

