function [dtModel, info] = train_decision_tree(XTrain, yTrain)
%TRAIN_DECISION_TREE Train a decision tree classifier with simple CV.
%
%   info: struct with fields
%       .hyperparamValues
%       .cvAccuracy
%       .bestHyperparam

    fprintf('\n=== Training Decision Tree ===\n');

    % Hyperparameter grid (tune MaxNumSplits)
    hyperparamValues = [20, 50, 100, 200, 400];

    kFolds = 5;
    cvAccuracy = zeros(size(hyperparamValues));

    for i = 1:numel(hyperparamValues)
        maxSplits = hyperparamValues(i);
        fprintf('  CV for MaxNumSplits = %d ...\n', maxSplits);

        params.maxNumSplits = maxSplits;
        cvAccuracy(i) = cross_validation(XTrain, yTrain, 'dt', params, kFolds);
    end

    % Choose best hyperparameter
    [~, bestIdx] = max(cvAccuracy);
    bestMaxSplits = hyperparamValues(bestIdx);
    fprintf('Best MaxNumSplits = %d (CV accuracy = %.3f)\n', ...
        bestMaxSplits, cvAccuracy(bestIdx));

    % Train final model on full training set
    dtModel = fitctree(XTrain, yTrain, 'MaxNumSplits', bestMaxSplits);

    info.modelName         = 'Decision Tree';
    info.hyperparamName    = 'MaxNumSplits';
    info.hyperparamValues  = hyperparamValues;
    info.cvAccuracy        = cvAccuracy;
    info.bestHyperparam    = bestMaxSplits;
end
