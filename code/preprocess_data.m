%PREPROCESS_DATA Basic preprocessing for MNIST pixels
function [XTrainProc, XTestProc] = preprocess_data(XTrain, XTest)

    fprintf('Preprocessing data (normalisation)...\n');

    XTrainProc = double(XTrain);
    XTestProc  = double(XTest);

    % scale pixels from [0,255] to [0,1]
    XTrainProc = XTrainProc / 255;
    XTestProc  = XTestProc  / 255;

    % visualize pixel intensity distribution
    figure;
    histogram(XTrainProc(:), 50);
    xlabel('Pixel intensity (rescaled to [0,1])');
    ylabel('Frequency');
    title('Distribution of rescaled pixel intensities');
    grid on;
    
    % normal standard distribution
    mu = mean(XTrainProc, 1);
    sigma = std(XTrainProc, 0, 1) + 1e-6;
    XTrainProc = (XTrainProc - mu) ./ sigma;
    XTestProc  = (XTestProc  - mu) ./ sigma;

    % visualize standardised pixel distribution
    figure;
    [f,xi] = ksdensity(XTrainProc(:));
    plot(xi,f,'LineWidth',2);
    xlabel('Pixel value (z-score)');
    ylabel('Density');
    title('Density of standardised pixel values');
    grid on;

end
