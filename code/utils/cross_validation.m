%CROSS_VALIDATION Simple k-fold cross-validation for DT / RF.
function meanAcc = cross_validation(X, y, modelType, params, kFolds)

    if nargin < 5
        kFolds = 5;
    end

    % Create partition for k-fold CV
    cv = cvpartition(y, 'KFold', kFolds);
    acc = zeros(kFolds, 1);

    for k = 1:kFolds
        % Get indices for this fold
        trainIdx = training(cv, k);
        testIdx  = test(cv, k);

        % Split data according to indices
        Xtr = X(trainIdx, :);
        ytr = y(trainIdx);
        Xte = X(testIdx, :);
        yte = y(testIdx);

        switch lower(modelType)
            case 'dt'
                % Train a classification tree with specified max splits
                mdl = fitctree(Xtr, ytr, ...
                    'MaxNumSplits', params.maxNumSplits);

                % Predict labels for validation fold
                yPred = predict(mdl, Xte);

            case 'rf'
                % Train a bagged ensemble for classification
                mdl = TreeBagger(params.numTrees, Xtr, ytr, ...
                    'Method', 'classification', 'OOBPrediction', 'Off');

                % Same prediction but treeBagger.predict often returns cell arrays -> convert to numeric
                yPred = predict(mdl, Xte);
                if iscell(yPred)
                    yPred = str2double(yPred);
                end

            otherwise
                error('Unknown modelType: %s', modelType);
        end

        % Compute fold accuracy
        acc(k) = mean(yPred == yte);
    end

    % Return mean accuracy across folds
    meanAcc = mean(acc);
end
