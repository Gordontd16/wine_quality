% *** Feature Selection Comparison ***

%AIM: use lasso regression to explore coefficients of features; we explore
%feature selection for Random Forest using the "predictorImportance" MATLAB 
%function.

%Clear workspace and Command window
clear; clc; close all;

%Load the dataset
data = readtable('winequality-white.csv', 'PreserveVariableNames', true);

%Define a new variable 'good_quality' for wines with quality >= 7.
data.good_quality = data.quality >= 7;

%Create X and Y matrices
X = table2array(data(:, 1:11));
y = logical(data.good_quality);

%Exploring Feature Selection with Logistic Regression using Lasso (L1 Norm)%

% Fitting Lasso Model using Cross Validation with different values of
% Lambda
rng default
[mdlLasso,FitInfo] = lassoglm(X, y, 'binomial', 'CV', 10, 'Standardize', true);

% Figure 1: Deviance Plot
lassoPlot(mdlLasso, FitInfo, 'PlotType', 'CV');
legend('show', 'Location', 'best') % show legend

% Figure 2: Trace Plot of coefficients found by Lasso for each predictor as a function of Lambda
lassoPlot(mdlLasso,FitInfo,'PlotType','Lambda','XScale','log');
ylabel('Coefficient')

% Find the number of nonzero coefficients at the Lambda value with minimum deviance plus one standard deviation point
idxLambda1SE = FitInfo.Index1SE;
B0 = mdlLasso(:, idxLambda1SE);
nonzeros = sum(B0 ~= 0); % Number of features with coefficient != 0

fprintf('The number of original features is %d.\n', length(B0))
fprintf('The number of features kept after Lasso Regularization is %d.\n', nonzeros)

%Figure 3: Lasso Coefficients
coef_table = array2table(abs(B0),'RowNames',data.Properties.VariableNames(1:11));
coef_table = sortrows(coef_table,'Var1');
figure('pos',[10 1000 500 400]);
Xbar = categorical(coef_table.Properties.RowNames);
Xbarstr = string(Xbar);
Xbar = reordercats(Xbar, Xbarstr);
Ybar = coef_table.Var1;
barh(Xbar, Ybar);
%title('Lasso Regression Coefficients');
ylabel('Coefficients');
xlabel('Predictors');
h = gca;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';
%The greater the absolute value of the coefficient is, the more significant 
%the predictor is in predicting good quality wine .
%The bar graph suggests that chlorides is the most important predictor, 
%followed by volatile acidity.

%Exploring Feature Selection with Random Forest%
mdlRF = fitcensemble(X,y, 'Method', 'Bag');
L = resubLoss(mdlRF);
%L is 0, which indicates that mdlRF is nearly perfect at classifying the training data.
%Unbiased Predictor Importance Estimates
imp = predictorImportance(mdlRF);

%Figure 4: Random Forest Feature Importance 
imptrans = imp.';
imp_table = array2table(imptrans,'RowNames',data.Properties.VariableNames(1:11));
imp_table = sortrows(imp_table,'imptrans');
figure('pos',[1000 1000 500 400]);
Xbar1 = categorical(imp_table.Properties.RowNames);
Xbarstr1 = string(Xbar1);
Xbar1 = reordercats(Xbar1, Xbarstr1);
Ybar1 = imp_table.imptrans;
barh(Xbar1, Ybar1);
%title('Random Forest Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
h = gca;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';
%Greater importance estimates indicate more important predictors.
%The bar graph suggests that density is the most important predictor, 
%followed by alcohol.

