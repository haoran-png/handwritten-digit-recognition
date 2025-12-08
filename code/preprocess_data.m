function [XTrainProc, XTestProc] = preprocess_data(XTrain, XTest)
%PREPROCESS_DATA Basic preprocessing for MNIST pixels.

    fprintf('Preprocessing data (normalisation)...\n');

    XTrainProc = double(XTrain);
    XTestProc  = double(XTest);

    % Pixels are typically 0–255 -> scale to [0,1]
    XTrainProc = XTrainProc / 255;
    XTestProc  = XTestProc  / 255;
    
    % standardize to zero mean, unit variance
    mu = mean(XTrainProc, 1);
    sigma = std(XTrainProc, 0, 1) + 1e-6;
    XTrainProc = (XTrainProc - mu) ./ sigma;
    XTestProc  = (XTestProc  - mu) ./ sigma;
end
