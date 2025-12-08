function metricsStruct = metrics(yTrue, yPred)
%METRICS Compute basic classification metrics for multi-class MNIST.

    % Ensure column vectors
    yTrue = yTrue(:);
    yPred = yPred(:);

    % Accuracy
    accuracy = mean(yTrue == yPred);

    % Confusion matrix
    classes = unique(yTrue);
    C = confusionmat(yTrue, yPred, 'Order', classes);

    % Per-class precision/recall/F1
    numClasses = numel(classes);
    precision  = zeros(numClasses, 1);
    recall     = zeros(numClasses, 1);
    f1         = zeros(numClasses, 1);

    for i = 1:numClasses
        tp = C(i, i);
        fp = sum(C(:, i)) - tp;
        fn = sum(C(i, :)) - tp;

        precision(i) = tp / max(tp + fp, eps);
        recall(i)    = tp / max(tp + fn, eps);
        f1(i)        = 2 * precision(i) * recall(i) / max(precision(i) + recall(i), eps);
    end

    metricsStruct = struct();
    metricsStruct.accuracy  = accuracy;
    metricsStruct.confusion = C;
    metricsStruct.classes   = classes;
    metricsStruct.precision = precision;
    metricsStruct.recall    = recall;
    metricsStruct.f1        = f1;
end
