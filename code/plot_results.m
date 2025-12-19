%PLOT_RESULTS Plot accuracy comparison and hyperparameter tuning curves.
function plot_results(dtResults, rfResults, dtInfo, rfInfo, accFigPath, hyperFigPath)

    % Accuracy comparison
    figure('Visible','off');
    modelNames = {dtResults.modelName, rfResults.modelName};
    testAcc    = [dtResults.testAcc,  rfResults.testAcc];

    bar(testAcc);
    set(gca, 'XTickLabel', modelNames);
    ylabel('Test accuracy');
    title('Decision Tree vs Random Forest on MNIST');

    for i = 1:numel(testAcc)
        text(i, testAcc(i), sprintf('%.3f', testAcc(i)), ...
            'HorizontalAlignment','center', 'VerticalAlignment','bottom');
    end

    % Save accuracy figure
    if ~isempty(accFigPath)
        [figDir, ~, ~] = fileparts(accFigPath);
        if ~exist(figDir, 'dir'); mkdir(figDir); end
        saveas(gcf, accFigPath);
    end
    close(gcf);

    % Hyperparameter curves
    figure('Visible','off');
    hold on;

    plot(dtInfo.hyperparamValues, dtInfo.cvAccuracy, '-o', 'DisplayName', dtInfo.modelName);
    plot(rfInfo.hyperparamValues, rfInfo.cvAccuracy, '-s', 'DisplayName', rfInfo.modelName);

    xlabel('Hyperparameter value');
    ylabel('CV accuracy');
    title('Hyperparameter Tuning');
    legend('Location','best');
    grid on;

    % Save hyperparameter tuning figure
    if ~isempty(hyperFigPath)
        [figDir, ~, ~] = fileparts(hyperFigPath);
        if ~exist(figDir, 'dir'); mkdir(figDir); end
        saveas(gcf, hyperFigPath);
    end
    close(gcf);
end
