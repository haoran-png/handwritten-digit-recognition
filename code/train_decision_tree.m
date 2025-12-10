%TRAIN_DECISION_TREE Train a decision tree classifier with simple CV
function [dtModel, info] = train_decision_tree(XTrain, yTrain)

    fprintf('\n=== Training Decision Tree ===\n');

    % Hyperparameter grid
    hyperparamValues = [100, 200, 400, 600, 800];

    kFolds = 5;
    cvAccuracy = zeros(size(hyperparamValues));

    % Time the cross validation
    tCV = tic;
    for i = 1:numel(hyperparamValues)
        maxSplits = hyperparamValues(i);
        fprintf('  CV for MaxNumSplits = %d ...\n', maxSplits);

        params.maxNumSplits = maxSplits;
        cvAccuracy(i) = cross_validation(XTrain, yTrain, 'dt', params, kFolds);
    end
    cvTimeSeconds = toc(tCV);

    % Choose best hyperparameter
    [~, bestIdx] = max(cvAccuracy);
    bestMaxSplits = hyperparamValues(bestIdx);
    fprintf('Best MaxNumSplits = %d (CV accuracy = %.3f)\n', ...
        bestMaxSplits, cvAccuracy(bestIdx));

    % Time the final training
    tTrain = tic;
    dtModel = fitctree(XTrain, yTrain, 'MaxNumSplits', bestMaxSplits);
    trainTimeSeconds = toc(tTrain);

    % Fill info structure
    info.modelName         = 'Decision Tree';
    info.hyperparamName    = 'MaxNumSplits';
    info.hyperparamValues  = hyperparamValues;
    info.cvAccuracy        = cvAccuracy;
    info.bestHyperparam    = bestMaxSplits;
    info.cvTimeSeconds     = cvTimeSeconds;
    info.trainTimeSeconds  = trainTimeSeconds;
end
