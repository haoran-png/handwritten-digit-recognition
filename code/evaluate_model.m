%EVALUATE_MODEL Evaluate model on train and test sets, save confusion matrix.
function results = evaluate_model(model, XTrain, yTrain, XTest, yTest, modelName, confusionFigPath)


    fprintf('\n=== Evaluating %s ===\n', modelName);

    % Predictions on train data set
    yPredTrain = local_predict(model, XTrain);
    trainMetrics = metrics(yTrain, yPredTrain);
    fprintf('%s - Train accuracy: %.4f\n', modelName, trainMetrics.accuracy);

    % Predictions on test data set
    yPredTest = local_predict(model, XTest);
    testMetrics = metrics(yTest, yPredTest);
    fprintf('%s - Test accuracy:  %.4f\n', modelName, testMetrics.accuracy);

    % Plot confusion matrix for test set
    figure('Visible','off');
    confusionchart(testMetrics.confusion, string(testMetrics.classes));
    title(sprintf('Confusion Matrix - %s (Test)', modelName));

    if ~isempty(confusionFigPath)
        [figDir, ~, ~] = fileparts(confusionFigPath);
        if ~exist(figDir, 'dir'); mkdir(figDir); end
        saveas(gcf, confusionFigPath);
    end
    close(gcf);

    % Package results
    results = struct();
    results.modelName    = modelName;
    results.trainAcc     = trainMetrics.accuracy;
    results.testAcc      = testMetrics.accuracy;
    results.metricsTest  = testMetrics;
    results.metricsTrain = trainMetrics;
end

% -------------------------------

% Local helper for predictions
function yPred = local_predict(model, X)
    if isa(model, 'TreeBagger')
        yPred = predict(model, X);
        if iscell(yPred)
            yPred = str2double(yPred);
        end
    else
        % Assume it's a ClassificationTree or similar
        yPred = predict(model, X);
    end
end
