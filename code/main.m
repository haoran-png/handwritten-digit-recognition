% IN3300 Project – Decision Tree vs Random Forest on MNIST
clear; clc; close all;
rng(42);

% filepath management
thisFile = mfilename('fullpath');
[thisDir,~,~] = fileparts(thisFile);

addpath(fullfile(thisDir, 'utils'));

% load_data.m
[XTrain, yTrain, XTest, yTest] = load_data();

% preprocess_data.m
[XTrainProc, XTestProc] = preprocess_data(XTrain, XTest);

% train_decision_tree.m
% train_random_forest.m
[dtModel, dtInfo] = train_decision_tree(XTrainProc, yTrain);
[rfModel, rfInfo] = train_random_forest(XTrainProc, yTrain);

% time for training and CV
fprintf('\n=== Training times ===\n');
fprintf('Decision Tree  - train: %.3f s, CV: %.3f s\n', ...
    dtInfo.trainTimeSeconds, dtInfo.cvTimeSeconds);
fprintf('Random Forest  - train: %.3f s, CV: %.3f s\n', ...
    rfInfo.trainTimeSeconds, rfInfo.cvTimeSeconds);

% print hyperparameter tuning results
fprintf('\n=== Hyperparameter Tuning Results ===\n');

fprintf('\nDecision Tree (MaxNumSplits):\n');
for i = 1:numel(dtInfo.hyperparamValues)
    fprintf('  MaxNumSplits = %d: CV Accuracy = %.4f\n', ...
        dtInfo.hyperparamValues(i), dtInfo.cvAccuracy(i));
end

fprintf('\nRandom Forest (NumTrees):\n');
for i = 1:numel(rfInfo.hyperparamValues)
    fprintf('  NumTrees = %d: CV Accuracy = %.4f\n', ...
        rfInfo.hyperparamValues(i), rfInfo.cvAccuracy(i));
end

% evaluate_model.m
dtResults = evaluate_model(dtModel, XTrainProc, yTrain, ...
                           XTestProc,  yTest,  'Decision Tree', ...
                           fullfile('results','confusion_dt.png'));

rfResults = evaluate_model(rfModel, XTrainProc, yTrain, ...
                           XTestProc,  yTest,  'Random Forest', ...
                           fullfile('results','confusion_rf.png'));

% plot_results.m
resultsDir = fullfile('results');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

plot_results(dtResults, rfResults, dtInfo, rfInfo, ...
    fullfile('results','accuracy_comparison.png'), ...
    fullfile('results','hyperparameter_plot.png'));
