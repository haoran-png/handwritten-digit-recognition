% IN3300 Project – Decision Tree vs Random Forest on MNIST

clear; clc; close all;
rng(42);                          % for reproducibility

% Figure out the folder where THIS file (main.m) lives
thisFile = mfilename('fullpath');
[thisDir,~,~] = fileparts(thisFile);

% Add utils folder (code/utils)
%addpath(fullfile(thisDir, 'utils'));

% ----------------------------------------------------------------------
% load_data.m
[XTrain, yTrain, XTest, yTest] = load_data();

% ----------------------------------------------------------------------
% preprocess_data.m
[XTrainProc, XTestProc] = preprocess_data(XTrain, XTest);

%----------------------------------------------------------------------
% train_decision_tree.m
% train_random_forest.m
[dtModel, dtInfo] = train_decision_tree(XTrainProc, yTrain);
[rfModel, rfInfo] = train_random_forest(XTrainProc, yTrain);

%----------------------------------------------------------------------
% evaluate_model.m
dtResults = evaluate_model(dtModel, XTrainProc, yTrain, ...
                           XTestProc,  yTest,  'Decision Tree', ...
                           fullfile('results','confusion_dt.png'));

rfResults = evaluate_model(rfModel, XTrainProc, yTrain, ...
                           XTestProc,  yTest,  'Random Forest', ...
                           fullfile('results','confusion_rf.png'));

%----------------------------------------------------------------------
% plot_results.m
resultsDir = fullfile('results');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

plot_results(dtResults, rfResults, dtInfo, rfInfo, ...
    fullfile('results','accuracy_comparison.png'), ...
    fullfile('results','hyperparameter_plot.png'));
