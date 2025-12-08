function meanAcc = cross_validation(X, y, modelType, params, kFolds)
%CROSS_VALIDATION Simple k-fold cross-validation for DT / RF.
%
%   modelType: 'dt' or 'rf'
%   params: struct with fields depending on modelType
%   Returns mean accuracy over k folds.

    if nargin < 5
        kFolds = 5;
    end

    cv = cvpartition(y, 'KFold', kFolds);
    acc = zeros(kFolds, 1);

    for k = 1:kFolds
        trainIdx = training(cv, k);
        testIdx  = test(cv, k);

        Xtr = X(trainIdx, :);
        ytr = y(trainIdx);
        Xte = X(testIdx, :);
        yte = y(testIdx);

        switch lower(modelType)
            case 'dt'
                mdl = fitctree(Xtr, ytr, ...
                    'MaxNumSplits', params.maxNumSplits);

                yPred = predict(mdl, Xte);

            case 'rf'
                mdl = TreeBagger(params.numTrees, Xtr, ytr, ...
                    'Method', 'classification', 'OOBPrediction', 'Off');

                yPred = predict(mdl, Xte);
                % TreeBagger returns cell array of char for classification
                if iscell(yPred)
                    yPred = str2double(yPred);
                end

            otherwise
                error('Unknown modelType: %s', modelType);
        end

        acc(k) = mean(yPred == yte);
    end

    meanAcc = mean(acc);
end
