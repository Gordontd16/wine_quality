% *** Performance Comparison ***

%AIM: Compare the performance of Logistic Regression and Random Forest in 
%terms of predictive power, training and prediction time ***

% Clear workspace and Command window
clear; clc; close all;

% Load the dataset
data = readtable('winequality-white.csv', 'PreserveVariableNames', true);

%load optimised hyperparameter values
load('optLR1.mat')
load('optRF1.mat')

% Define a new variable 'good_quality' for wines with quality >= 7.
data.good_quality = data.quality >= 7;

%Create X and Y matrices
X = table2array(data(:, 1:11));
y = logical(data.good_quality);

% Model Training using optimised hyperparameters
rng default
mdlLRCV = fitclinear(X,y, 'Learner', 'logistic', 'Lambda', BestLambda1, 'Regularization',...
    BestReg1, 'CrossVal', 'On');
t = templateTree('MinLeafSize', BestMLS1, 'Reproducible',true); % For reproducibiliy of random predictor selections
mdlRFCV = fitcensemble(X,y, 'Method', 'Bag', 'NumLearningCycles', BestNLC1,...
    'Learners',t, 'CrossVal', 'On');

% Model Predictions & Confusion Matrices
    %Logistic Regression%
[ClassLR, ScoreLR] = kfoldPredict(mdlLRCV);
figure('pos',[10 1000 500 400]);
cmLR = confusionchart(y,ClassLR);
%ClassLR are the predictions of the Logistic Regression model and yTest are
%the true values
tnLR = sum((ClassLR == 0) & (y == 0));
tpLR = sum((ClassLR == 1) & (y == 1));
fpLR = sum((ClassLR == 1) & (y == 0));
fnLR = sum((ClassLR == 0) & (y == 1));
%Accuracy for LR
accLR = (tnLR + tpLR) / (tnLR+tpLR+fpLR+fnLR);
%Calculating F1 Score for LR
precisionLR = tpLR / (tpLR + fpLR);
recallLR = tpLR / (tpLR + fnLR);
F1LR = (2 * precisionLR * recallLR) / (precisionLR + recallLR);
    %Random Forest
[ClassRF, ScoreRF] = kfoldPredict(mdlRFCV);
figure('pos',[1000 1000 500 400]);
cmRF = confusionchart(y,ClassRF);
%ClassRF are the predictions of the Random Forest model and yTest are
%the true values
tnRF = sum((ClassRF == 0) & (y == 0));
tpRF = sum((ClassRF == 1) & (y == 1));
fpRF = sum((ClassRF == 1) & (y == 0));
fnRF = sum((ClassRF == 0) & (y == 1));
%Accuracy for RF
accRF = (tnRF + tpRF) / (tnRF+tpRF+fpRF+fnRF);
%Calculating F1 Score for RF
precisionRF = tpRF / (tpRF + fpRF);
recallRF = tpRF / (tpRF + fnRF);
F1RF = (2 * precisionRF * recallRF) / (precisionRF + recallRF);

% ROC Curve
[XLR, YLR, TLR, AUCLR] = perfcurve(y,ScoreLR(:,2),'true');
[XRF, YRF, TRF, AUCRF] = perfcurve(y,ScoreRF(:,2),'true');

% Model Training/Prediction Time Performance
LRfitHandle = @() fitclinear(X,y, 'Learner', 'logistic', 'Lambda',...
    BestLambda1, 'Regularization', BestReg1);
LRpredHandle = @() kfoldPredict(mdlLRCV);
RFfitHandle = @() fitcensemble(X,y, 'Method', 'Bag', 'NumLearningCycles',...
    BestNLC1, 'Learners',t);
RFpredHandle = @() kfoldPredict(mdlRFCV);

timeLRfit = timeit(LRfitHandle);
timeLRpred = timeit(LRpredHandle);
timeRFfit = timeit(RFfitHandle);
timeRFpred = timeit(RFpredHandle);

% Printing Performance
    % Prediction Performance
fprintf('AUC Prediction Performance -----------\n')
fprintf('Logistic Regression : %.3f\n', AUCLR)
fprintf('Random Forest       : %.3f\n\n', AUCRF)
    % Time Performance
fprintf('Time Performance (seconds) -----------\n')
fprintf('      Training Time Performance ------\n')
fprintf('Logistic Regression : %.3f\n', timeLRfit)
fprintf('Random Forest       : %.3f\n', timeRFfit)
fprintf('      Prediction Time Performance ----\n')
fprintf('Logistic Regression : %.3f\n', timeLRpred)
fprintf('Random Forest       : %.3f\n', timeRFpred)

% Visualise ROC Curve
figure
plot(XLR, YLR)
hold on
plot(XRF, YRF)
%title('ROC Curve Comparison : Logistic Regression vs Random Forest')
xlabel('False Positive Rate'); ylabel('True Positive Rate');
legend( 'Logistic Regression', 'Random Forest')
