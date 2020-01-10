% *** Optimise models ***

%AIM: carry out experiments to find the optimised hyperparameters for both
%algorithms aiming to minimise the cross-validation loss (error) by varying
%the parameters and comparing Bayesian Optimisation to grid search; for
%Logistic Regression we are aiming to find the optimised lambda
%and the best regularization; for Random Forest we are searching for the
%best number of learning cycles (trees) and the best minimum leaf size.

% Clear workspace and Command window
clear; clc; close all;

% Load the dataset
data = readtable('winequality-white.csv', 'PreserveVariableNames', true);

% Define a new variable 'good_quality' for wines with quality >= 7.
data.good_quality = data.quality >= 7;

%Create X and Y matrices
X = table2array(data(:, 1:11));
y = data.good_quality;

%Optimise Linear Classifier with Bayesian Optimization
rng default % Set the seed for reproducibility
tic
mdlLR1 = fitclinear(X,y, 'Learner', 'logistic', 'OptimizeHyperparameters',...
    {'Lambda', 'Regularization'},'HyperparameterOptimizationOptions',...
    struct('Kfold', 10, 'AcquisitionFunctionName','expected-improvement-plus'));
%title('Logistic Regression Hyperparameter Tuning with Bayesian Optimization');
toc
BestLambda1 = mdlLR1.Lambda;
BestReg1 = 'ridge';
save ('optLR1.mat', 'BestLambda1', 'BestReg1')

% %Optimise Linear Classifier with grid search
% rng default % Set the seed for reproducibility
% tic
% mdlLR2 = fitclinear(X,y, 'Learner', 'logistic', 'OptimizeHyperparameters',...
%     {'Lambda', 'Regularization'},'HyperparameterOptimizationOptions',...
%     struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 30, 'Kfold', 10, 'AcquisitionFunctionName','expected-improvement-plus'));
% title('Logistic Regression Hyperparameter Tuning with grid search');
% toc
% BestLambda2 = mdlLR2.Lambda;
% BestReg2 = 'ridge';
%save ('optLR2.mat', 'BestLambda2', 'BestReg2')

%Optimise Classification Ensemble with Bayesian Optimization
rng default % Set the seed for reproducibility
tic
t = templateTree('Reproducible',true); % For reproducibility of random predictor selections
mdlRF1 = fitcensemble(X,y, 'Method', 'Bag', 'Learners',t, 'OptimizeHyperparameters',...
    {'NumLearningCycles', 'MinLeafSize'},'HyperparameterOptimizationOptions',...
    struct('Kfold', 10, 'AcquisitionFunctionName','expected-improvement-plus'));
%title('Random Forest Hyperparameter Tuning with Bayesian Optimization');
toc
BestNLC1 = mdlRF1.NumTrained;
BestMLS1 = 1;
save ('optRF1.mat', 'BestNLC1', 'BestMLS1')

% %Optimise Classification Ensemble with grid search
% rng default % Set the seed for reproducibility
% tic
% t = templateTree('Reproducible',true); % For reproducibility of random predictor selections
% mdlRF2 = fitcensemble(X,y, 'Method', 'Bag', 'Learners',t, 'OptimizeHyperparameters',...
%     {'NumLearningCycles', 'MinLeafSize'},'HyperparameterOptimizationOptions',...
%     struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 30,'Kfold', 10, 'AcquisitionFunctionName','expected-improvement-plus'));
% title('Random Forest Hyperparameter Tuning with grid search');
% toc
% BestNLC2 = mdlRF2.NumTrained;
% BestMLS2 = 1;
%save ('optRF2.mat', 'BestNLC2', 'BestMLS2')