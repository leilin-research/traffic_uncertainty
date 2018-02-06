function [objVal_train,objVal_test,TY,flag,InputWeight,biasofHiddenNeurons,OutputWeight,A1,A,S,psoIterRecord,elmobjVal_test,elmflag,elmA1,elmA,elmS]= elm_pi( training,testing,No_of_Output,NumberofHiddenNeurons, ActivationFunction,popNum,iteNum,w,c1,c2,fai,conInterval,speed,lo,m1,m2,w1,w2, ol)
%ActivationFunction: sigmoid sine hardlim
    %%%%%%%%%%% Load training dataset
    train_data=training;
    T=train_data(:,1:No_of_Output)';
    P=train_data(:,No_of_Output+1:size(train_data,2))';
    clear train_data;                                   %   Release raw training data array
    
    %%%%%%%%%%% Load testing dataset
    test_data=testing;
    TV.T=test_data(:,1:No_of_Output)';
    TV.P=test_data(:,No_of_Output+1:size(test_data,2))';
    clear test_data;
    
    NumberofTrainingData=size(P,2); 
    NumberofTestingData=size(TV.P,2);
    NumberofInputNeurons=size(P,1);% transpose
    
    %%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
    InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1; % rand generates a value between 0 and 1, times 2, minus 1, generate [-1,1]
    biasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
    ind=ones(1,NumberofTrainingData);
    biasMatrix=biasofHiddenNeurons(:,ind);
    
    tempH=InputWeight*P;
    tempH=tempH+biasMatrix; % each column of tempH is the input of the activation fucntion of the hidden layer
    
    %%%% the activation fuction
    switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = hardlim(tempH);            
        %%%%%%%% More activation functions can be added here                
    end
    clear tempH; 
    
    %%%%%%%%%%% Calculate an initial output weights OutputWeight (beta_i)
    OutputWeight=pinv(H') * T';
    
    %%%%%%%%%%% Calculate the prediction using ELM directly
    tempH_test=InputWeight*TV.P;
    clear TV.P;             %   Release input of testing data             
    ind=ones(1,NumberofTestingData);
    BiasMatrix=biasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
    tempH_test=tempH_test + BiasMatrix;
    switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
        %%%%%%%% More activation functions can be added here        
    end
    TY=(H_test' * OutputWeight)';   
    [ elmobjVal_test,elmflag,elmA1,elmA, elmS,elmTY] = elm_calObject( TY,TV.T(1,:)/(1-lo), conInterval,m1,m2, w1, w2, ol);
    
    %%%%%%%%%%% Calculate the output weights throught PSO algorithm
    [OutputWeight,objVal_train,psoIterRecord]=pso(OutputWeight,H,popNum,iteNum,T,conInterval,speed,w,c1,c2,fai,lo,m1,m2, w1, w2, ol);
   
    %%% calculate the prediction for the testing dataset
    tempH_test=InputWeight*TV.P;
    clear TV.P;             %   Release input of testing data             
    ind=ones(1,NumberofTestingData);
    BiasMatrix=biasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
    tempH_test=tempH_test + BiasMatrix;
    switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
        %%%%%%%% More activation functions can be added here        
    end
    TY=(H_test' * OutputWeight)';   
    [ objVal_test,flag,A1,A,S,TY] = elm_calObject( TY,TV.T(1,:)/(1-lo), conInterval,m1,m2, w1, w2, ol);
    
%     pre=mean(abs(TY-TV.T)./TV.T);
end

