% LOAD_DATA Reads MNIST IDX files into MATLAB matrices.
function [XTrain, yTrain, XTest, yTest] = load_data()

    % File paths
    thisDir = fileparts(mfilename('fullpath'));
    baseDir = fullfile(thisDir, '..', 'data');

    trainImagesFile = fullfile(baseDir, 'train-images.idx3-ubyte');
    trainLabelsFile = fullfile(baseDir, 'train-labels.idx1-ubyte');
    testImagesFile  = fullfile(baseDir, 't10k-images.idx3-ubyte');
    testLabelsFile  = fullfile(baseDir, 't10k-labels.idx1-ubyte');

    % Verify files exist
    assert(isfile(trainImagesFile), 'File not found: %s', trainImagesFile);
    assert(isfile(trainLabelsFile), 'File not found: %s', trainLabelsFile);
    assert(isfile(testImagesFile),  'File not found: %s', testImagesFile);
    assert(isfile(testLabelsFile),  'File not found: %s', testLabelsFile);

    % Lode training data
    XTrain = load_images(trainImagesFile);
    yTrain = load_labels(trainLabelsFile);

    % Load test data
    XTest  = load_images(testImagesFile);
    yTest  = load_labels(testLabelsFile);

    fprintf('Loaded MNIST: %d train samples, %d test samples.\n', ...
        size(XTrain,1), size(XTest,1));
end

% -------------------------------

% Helper function to load images
function images = load_images(filename)
    fid = fopen(filename,'rb');
    if fid < 0, error('Could not open %s', filename); end

    magic = fread(fid,1,'int32',0,'ieee-be');
    if magic ~= 2051
        error('Invalid magic number in MNIST image file %s',filename);
    end

    numImages = fread(fid,1,'int32',0,'ieee-be');
    numRows   = fread(fid,1,'int32',0,'ieee-be');
    numCols   = fread(fid,1,'int32',0,'ieee-be');

    rawData = fread(fid, numImages*numRows*numCols, 'uint8');
    fclose(fid);

    rawData = reshape(rawData, numRows*numCols, numImages)';
    images = double(rawData);  % convert to double
end

% -------------------------------

% Helper function to load labels
function labels = load_labels(filename)
    fid = fopen(filename,'rb');
    if fid < 0, error('Could not open %s', filename); end

    magic = fread(fid,1,'int32',0,'ieee-be');
    if magic ~= 2049
        error('Invalid magic number in MNIST label file %s',filename);
    end

    numLabels = fread(fid,1,'int32',0,'ieee-be');
    labels = fread(fid, numLabels, 'uint8');
    fclose(fid);
end
