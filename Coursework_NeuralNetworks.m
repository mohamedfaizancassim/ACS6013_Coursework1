%=====================================
%   Data Aqusition and Clean Up
%=====================================
housing_data=readtable("CW_dataset.csv");

MSSubClass_num=grp2idx(housing_data.MSSubClass);
MSZoning_num=grp2idx(housing_data.MSZoning);

LotFrontage_num=housing_data.LotFrontage; %Already Numerical
LotFrontage_num=normalize(LotFrontage_num,'range'); %Normalization
LotArea_num=housing_data.LotArea; %Already Nuerical
LotArea_num=normalize(LotArea_num,'range'); %Normalization

LotShape_num=grp2idx(housing_data.LotShape);
BldgType_num=grp2idx(housing_data.BldgType);
HouseStyle_num=grp2idx(housing_data.HouseStyle);
OverallQual_num=grp2idx(housing_data.OverallQual);
OverallCond_num=grp2idx(housing_data.OverallCond);
Foundation_num=grp2idx(housing_data.Foundation);
BsmtQual_num=grp2idx(housing_data.BsmtQual);
BsmtCond_num=grp2idx(housing_data.BsmtCond);
Heating_num=grp2idx(housing_data.Heating);
HeatingQc_num=grp2idx(housing_data.HeatingQC);
CentralAir_num=grp2idx(housing_data.CentralAir);
Electrical_num=grp2idx(housing_data.Electrical);
KitchenQual_num=grp2idx(housing_data.KitchenQual);

Fireplaces_num=housing_data.Fireplaces; %Already Numerical
Fireplaces_num=normalize(Fireplaces_num,'range'); %Normalization

FireplacesQu_num=grp2idx(housing_data.FireplaceQu);
GarageQual_num=grp2idx(housing_data.GarageQual);
GarageCond_num=grp2idx(housing_data.GarageCond);
PoolQC_num=grp2idx(housing_data.PoolQC);
Fence_num=grp2idx(housing_data.Fence);
SaleCondition_num=grp2idx(housing_data.SaleCondition);


%=====================================
%   Feature Engineering
%=====================================

%Input Variable
features=table(MSSubClass_num,MSZoning_num,LotFrontage_num,LotArea_num, ...
    LotShape_num,BldgType_num,HouseStyle_num,OverallQual_num,OverallCond_num, ...
    Foundation_num,BsmtQual_num,BsmtCond_num,Heating_num,HeatingQc_num,CentralAir_num, ...
    Electrical_num,KitchenQual_num,Fireplaces_num,FireplacesQu_num,GarageQual_num,GarageCond_num, ...
    PoolQC_num,Fence_num,SaleCondition_num);
features=table2array(features);
features=transpose(features);

%Target Variable
target=log(housing_data.SalePrice);
%The target variable must be transposed.
target=transpose(target); 


%=====================================
%   Neural Network
%=====================================

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

%Choose the number of hidden layers.
hiddenLayerSize=26;
net=fitnet(hiddenLayerSize,trainFcn);

%Setup Train, Validation and Testing Data Ratios
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

%Train the network
[net,tr] = train(net,features,target);

% Test the Network
y = net(features);
e = gsubtract(target,y);
performance = perform(net,target,y)

% View the Network
view(net)


