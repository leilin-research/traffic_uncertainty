% res columns:
% objVal = reliability + normalsharp
% reliability is abs(PICP-PINP)
% noabs_reli = PICP-PINP
% normalisharp sharpness
% mean() -MPIL

% off-line arma
a=xlsread('traffic_900.csv');
m1=max(a);
m2=min(a);
b=xlsread('ARMA.csv');
res=[];
target=a(601:900);
b90=[b(2:301,3), b(2:301,6)];
b95=[b(2:301,4), b(2:301,7)];
b99=[b(2:301,5), b(2:301,8)];
[ objVal,reliability,noabs_reli, normalsharp] = calObject( b90,target,m1,m2,6,0.1, 0.90);
res=[res;[objVal, reliability,noabs_reli,normalsharp, mean(b90(:,2)-b90(:,1))]];
[ objVal,reliability,noabs_reli,normalsharp] = calObject( b95,target,m1,m2,11,0.1, 0.95);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b95(:,2)-b95(:,1))]];
[ objVal,reliability,noabs_reli,normalsharp] = calObject( b99,target,m1,m2,12,0.1, 0.99);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b99(:,2)-b99(:,1))]];

% kalman filter
a=xlsread('traffic_900.csv');
b=xlsread('Kalman.csv');
target=a(601:900);
b90=[b(602:901,3), b(602:901,4)];
b95=[b(602:901,5), b(602:901,6)];
b99=[b(602:901,7), b(602:901,8)];

[ objVal,reliability,noabs_reli,normalsharp] = calObject( b90,target,m1,m2,6,0.1, 0.90);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b90(:,2)-b90(:,1))]];
[ objVal,reliability,noabs_reli,normalsharp] = calObject( b95,target,m1,m2,11,0.1, 0.95);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b95(:,2)-b95(:,1))]];
[ objVal,reliability,noabs_reli,normalsharp] = calObject( b99,target,m1,m2,12,0.1, 0.99);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b99(:,2)-b99(:,1))]];

%spectral analysis + arma + gjr-garch -Zhang
a=xlsread('traffic_900.csv');
b=xlsread('zhang_2014.csv');
target=a(601:900);
b90=[b(2:301,3), b(2:301,4)];
b95=[b(2:301,5), b(2:301,6)];
b99=[b(2:301,7), b(2:301,8)];

[ objVal,reliability,noabs_reli,normalsharp] = calObject( b90,target,m1,m2,6,0.1, 0.90);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b90(:,2)-b90(:,1))]];
[ objVal,reliability,noabs_reli,normalsharp] = calObject( b95,target,m1,m2,11,0.1, 0.95);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b95(:,2)-b95(:,1))]];
[ objVal,reliability,noabs_reli,normalsharp] = calObject( b99,target,m1,m2,12,0.1, 0.99);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b99(:,2)-b99(:,1))]];

%kf for arma + garch - Williams
a=xlsread('traffic_900.csv');
b=xlsread('guo_2014.csv');
target=a(601:900);
b90=[b(2:301,3), b(2:301,4)];
b95=[b(2:301,5), b(2:301,6)];
b99=[b(2:301,7), b(2:301,8)];

[ objVal,reliability,noabs_reli,normalsharp] = calObject( b90,target,m1,m2,6,0.1, 0.90);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b90(:,2)-b90(:,1))]];
[ objVal,reliability,noabs_reli,normalsharp] = calObject( b95,target,m1,m2,11,0.1, 0.95);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b95(:,2)-b95(:,1))]];
[ objVal,reliability,noabs_reli,normalsharp] = calObject( b99,target,m1,m2,12,0.1, 0.99);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b99(:,2)-b99(:,1))]];


%improved online pso-elm
a=xlsread('traffic_900.csv');
b=xlsread('Improved_PSO_ELM.csv');
target=a(601:900);
b90=[b(2:301,1), b(2:301,2)];
b95=[b(2:301,3), b(2:301,4)];
b99=[b(2:301,5), b(2:301,6)];

[ objVal,reliability,noabs_reli,normalsharp] = calObject( b90,target,m1,m2,6,0.1, 0.90);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b90(:,2)-b90(:,1))]];
[ objVal,reliability,noabs_reli,normalsharp] = calObject( b95,target,m1,m2,11,0.1, 0.95);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b95(:,2)-b95(:,1))]];
[ objVal,reliability,noabs_reli,normalsharp] = calObject( b99,target,m1,m2,12,0.1, 0.99);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b99(:,2)-b99(:,1))]];

%improved online pso-elm
a=xlsread('traffic_900.csv');
b=xlsread('PSO_ELM.csv');
target=a(601:900);
b90=[b(2:301,1), b(2:301,2)];
b95=[b(2:301,3), b(2:301,4)];
b99=[b(2:301,5), b(2:301,6)];

[ objVal,reliability,noabs_reli,normalsharp] = calObject( b90,target,m1,m2,6,0.1, 0.90);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b90(:,2)-b90(:,1))]];
[ objVal,reliability,noabs_reli,normalsharp] = calObject( b95,target,m1,m2,11,0.1, 0.95);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b95(:,2)-b95(:,1))]];
[ objVal,reliability,noabs_reli,normalsharp] = calObject( b99,target,m1,m2,12,0.1, 0.99);
res=[res;[objVal,reliability,noabs_reli,normalsharp, mean(b99(:,2)-b99(:,1))]];

% objVal = reliability + normalsharp
% reliability is abs(PICP-PINP)
% noabs_reli = PICP-PINP
% normalisharp sharpness
% mean() -MPIL
