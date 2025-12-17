%TRAIN_RANDOM_FOREST Train a Random Forest (bagged trees) with simple CV
function [rfModel, info] = train_random_forest(XTrain, yTrain)

    fprintf('\n=== Training Random Forest ===\n');

    % Hyperparameter grid
    numTreesGrid = [50, 100, 200];
    kFolds = 5;
    cvAccuracy = zeros(size(numTreesGrid));

    % Time the cross validation
    tCV = tic;
    for i = 1:numel(numTreesGrid)
        params.numTrees = numTreesGrid(i);
        fprintf('  CV for NumTrees = %d ...\n', params.numTrees);
        cvAccuracy(i) = cross_validation(XTrain, yTrain, 'rf', params, kFolds);
    end
    cvTimeSeconds = toc(tCV);

    % Choose best hyperparameter
    [~, bestIdx] = max(cvAccuracy);
    bestNumTrees = numTreesGrid(bestIdx);
    fprintf('Best NumTrees = %d (CV accuracy = %.3f)\n', ...
        bestNumTrees, cvAccuracy(bestIdx));

    % Time the final training
    tTrain = tic;
    rfModel = TreeBagger(bestNumTrees, XTrain, yTrain, ...
                         'Method', 'classification', ...
                         'OOBPrediction', 'On');
    trainTimeSeconds = toc(tTrain);

    % Fill info structure
    info.modelName         = 'Random Forest';
    info.hyperparamName    = 'NumTrees';
    info.hyperparamValues  = numTreesGrid;
    info.cvAccuracy        = cvAccuracy;
    info.bestHyperparam    = bestNumTrees;
    info.cvTimeSeconds     = cvTimeSeconds;
    info.trainTimeSeconds  = trainTimeSeconds;
end
