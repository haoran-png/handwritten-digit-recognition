function [rfModel, info] = train_random_forest(XTrain, yTrain)
%TRAIN_RANDOM_FOREST Train a Random Forest (bagged trees) with simple CV.
%
%   info: struct similar to train_decision_tree

    fprintf('\n=== Training Random Forest ===\n');

    numTreesGrid = [50, 100, 200];
    kFolds = 5;
    cvAccuracy = zeros(size(numTreesGrid));

    for i = 1:numel(numTreesGrid)
        params.numTrees = numTreesGrid(i);
        fprintf('  CV for NumTrees = %d ...\n', params.numTrees);
        cvAccuracy(i) = cross_validation(XTrain, yTrain, 'rf', params, kFolds);
    end

    [~, bestIdx] = max(cvAccuracy);
    bestNumTrees = numTreesGrid(bestIdx);
    fprintf('Best NumTrees = %d (CV accuracy = %.3f)\n', ...
        bestNumTrees, cvAccuracy(bestIdx));

    % Train final model on full training set
    rfModel = TreeBagger(bestNumTrees, XTrain, yTrain, ...
                         'Method', 'classification', ...
                         'OOBPrediction', 'On');

    info.modelName         = 'Random Forest';
    info.hyperparamName    = 'NumTrees';
    info.hyperparamValues  = numTreesGrid;
    info.cvAccuracy        = cvAccuracy;
    info.bestHyperparam    = bestNumTrees;
end
