housing_data=readtable("CW_dataset.csv");


%Fillout Missing Data for Lot Frontage
TF=ismissing(housing_data.LotFrontage);
TF_index=find(~TF);
A=housing_data.LotFrontage(TF_index);
LotFrontage_median=median(A);
housing_data.LotFrontage=fillmissing(housing_data.LotFrontage,'constant',LotFrontage_median);

MSSubClass_Categorical=categorical(housing_data.MSSubClass);
MSZoning_Categorical=categorical(housing_data.MSZoning);
LotFrontage_Categorical=categorical(housing_data.LotFrontage);
LotArea_Categorical=categorical(housing_data.LotArea);
LotShape_Categorical=categorical(housing_data.LotShape);
BldgType_Categorical=categorical(housing_data.BldgType);
HouseStlye_Categorical=categorical(housing_data.HouseStyle);
OverallQual_Categorical=categorical(housing_data.OverallQual);
OverallCond_Categorical=categorical(housing_data.OverallCond);
Foundation_Categorical=categorical(housing_data.Foundation);
BsmtQual_Categorical=categorical(housing_data.BsmtQual);
BsmtCond_Categorical=categorical(housing_data.BsmtCond);
Heating_Categorical=categorical(housing_data.Heating);
HeatingQC_Categorical=categorical(housing_data.HeatingQC);
CentralAir_Categorical=categorical(housing_data.CentralAir);
Electrical_Categorical=categorical(housing_data.Electrical);
KitchenQual_Categorical=categorical(housing_data.KitchenQual);
FireplaceQu_Categorical=categorical(housing_data.FireplaceQu);
GarageQual_Categorical=categorical(housing_data.GarageQual);
GarageCond_Categorical=categorical(housing_data.GarageCond);
PoolQC_Categorical=categorical(housing_data.PoolQC);
Fence_Categorical=categorical(housing_data.Fence);
SaleCondition_Categorical=categorical(housing_data.SaleCondition);

%Features
X=table(MSSubClass_Categorical,MSZoning_Categorical,LotFrontage_Categorical,LotArea_Categorical,LotShape_Categorical,BldgType_Categorical,HouseStlye_Categorical, ...
    HouseStlye_Categorical,OverallQual_Categorical,OverallCond_Categorical,Foundation_Categorical,BsmtQual_Categorical,BsmtCond_Categorical, ...
    Heating_Categorical,HeatingQC_Categorical,CentralAir_Categorical,Electrical_Categorical,KitchenQual_Categorical,FireplaceQu_Categorical, ...
    PoolQC_Categorical,Fence_Categorical,SaleCondition_Categorical);

%Target
target=log(housing_data.SalePrice);


    
countLevels=@(X)numel(categories(categorical(X)));
numLevels=varfun(countLevels,X,'OutputFormat','uniform');


figure
bar(numLevels)
title('Number of Levels Among Predictors')
xlabel('Predictor variable')
ylabel('Number of levels')
h = gca;
h.XTickLabel = X.Properties.VariableNames(1:end-1);
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

nX=size(X);
nX=nX(1);
RMSE_Vals=[]
for perc=1:1:100
    cx=X(1:floor(nX*(perc/100)),:);
    t = templateTree('NumVariablesToSample','all',...
        'PredictorSelection','interaction-curvature','Surrogate','on');
    
    rng(1); % For reproducibility
    Mdl = fitrensemble(cx,target(1:nX*(perc/100),:),'Method','Bag','NumLearningCycles',200, ...
        'Learners',t);
    yHat=oobPredict(Mdl);
    R2=corr(Mdl.Y,yHat)^2;
    Loss=loss(Mdl,cx,target(1:nX*(perc/100),:));
    RMSE=sqrt(mean(Mdl.Y-yHat).^2);
    RMSE_Vals(end+1,:)=[perc,RMSE,R2];
end

plot(RMSE_Vals(:,1),RMSE_Vals(:,2));
hold on
plot(RMSE_Vals(:,1),RMSE_Vals(:,3));
title("Learning Curve");
legend("Training Data","Testing Data")
xlabel("Data %");
ylabel("RMSE");


