% MAIN_TEST – test-only run for video (loads saved models, evaluates on test set)
clear; clc; close all;
rng(42);

% Path setup
thisFile = mfilename('fullpath');
[thisDir,~,~] = fileparts(thisFile);

addpath(thisDir);
addpath(fullfile(thisDir, 'utils'));

projectDir = fileparts(thisDir);

% Load data + preprocess
[XTrain, yTrain, XTest, yTest] = load_data();
[XTrainProc, XTestProc] = preprocess_data(XTrain, XTest);

% Load pre-trained models and info
modelsDir = fullfile(projectDir, 'models');
load(fullfile(modelsDir, 'dt_model.mat'), 'dtModel');
load(fullfile(modelsDir, 'rf_model.mat'), 'rfModel');

load(fullfile(modelsDir, 'dt_info.mat'), 'dtInfo');
load(fullfile(modelsDir, 'rf_info.mat'), 'rfInfo');

% Output folder at project root
outDir = fullfile(projectDir, 'results', 'code_generated');
if ~exist(outDir, 'dir'); mkdir(outDir); end

% Evaluate
dtResults = evaluate_model(dtModel, XTrainProc, yTrain, XTestProc, yTest, ...
    'Decision Tree', fullfile(outDir, 'confusion_dt.png'));

rfResults = evaluate_model(rfModel, XTrainProc, yTrain, XTestProc, yTest, ...
    'Random Forest', fullfile(outDir, 'confusion_rf.png'));

% Plot summary figures (accuracy + hyperparameter curves)
plot_results(dtResults, rfResults, dtInfo, rfInfo, ...
    fullfile(outDir, 'accuracy_comparison.png'), ...
    fullfile(outDir, 'hyperparameter_plot.png'));
