clear, clc;

%%  read data files
data_path = 'data/MNIST'; 
train_images_file = 'train-images.idx3-ubyte';
train_labels_file = 'train-labels.idx1-ubyte';
test_images_file = 't10k-images.idx3-ubyte';
test_labels_file = 't10k-labels.idx1-ubyte';

% Function to read MNIST data
function [images, labels] = readMNIST(image_file, label_file)
    % Check if files exist
    if ~exist(image_file, 'file')
        error('Image file not found: %s', image_file);
    end
    if ~exist(label_file, 'file')
        error('Label file not found: %s', label_file);
    end

    % Read images
    fid = fopen(image_file, 'r');
    if fid == -1
        error('Could not open image file: %s', image_file);
    end
    magic = fread(fid, 1, 'int32', 'b');
    num_images = fread(fid, 1, 'int32', 'b');
    num_rows = fread(fid, 1, 'int32', 'b');
    num_cols = fread(fid, 1, 'int32', 'b');
    images = fread(fid, inf, 'unsigned char');
    images = reshape(images, num_cols, num_rows, num_images);
    images = permute(images, [2 1 3]);
    fclose(fid);
    
    % Read labels
    fid = fopen(label_file, 'r');
    magic = fread(fid, 1, 'int32', 'b');
    num_labels = fread(fid, 1, 'int32', 'b');
    labels = fread(fid, inf, 'unsigned char');
    fclose(fid);
end

% Load data - Update paths to include the full path to your MNIST folder
[train_images, train_labels] = readMNIST(...
    fullfile(data_path, train_images_file),...
    fullfile(data_path, train_labels_file));
[test_images, test_labels] = readMNIST(...
    fullfile(data_path, test_images_file),...
    fullfile(data_path, test_labels_file));

%% Extract digits 0 and 1
% Training set
train_idx = (train_labels == 0 | train_labels == 1);
train_features = train_images(:,:,train_idx);
train_labels = train_labels(train_idx);

% Test set
test_idx = (test_labels == 0 | test_labels == 1);
test_features = test_images(:,:,test_idx);
test_labels = test_labels(test_idx);

% image size
[h, w, ~] = size(train_images); 
image_size = [h, w]; 

%% Reshape data
% Convert images from 3D to 2D matrix (samples × features)
train_features = reshape(train_features, [], size(train_features, 3))';
test_features = reshape(test_features, [], size(test_features, 3))';

%% Normalize data
% Scale pixel values to range [0,1]
train_features = double(train_features) / 255;
test_features = double(test_features) / 255;

%% Save processed data
save(fullfile(data_path, 'data.mat'), 'train_features', 'train_labels', 'test_features', 'test_labels', 'image_size');

%% Display dataset information
fprintf('Training set size: %d samples\n', size(train_features, 1));
fprintf('Test set size: %d samples\n', size(test_features, 1));
fprintf('Number of features per sample: %d\n', size(train_features, 2));