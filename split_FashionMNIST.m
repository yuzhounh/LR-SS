% Define dataset name as a variable
data_path = 'data/FashionMNIST';  % data/MNIST, data/FashionMNIST

% Load the processed data
load(fullfile(data_path, 'data.mat'));

% Combine training and test sets
all_features = [train_features; test_features];
all_labels = [train_labels; test_labels];

% Number of samples for new training set
n_train = 1000;

% Use crossvalind for stratified sampling
rng(42);  % Set random seed for reproducibility
[train_idx, test_idx] = crossvalind('HoldOut', all_labels, 1 - n_train/length(all_labels));

% Convert logical indices to numeric indices
train_idx = find(train_idx);
test_idx = find(test_idx);

% Split the data
train_features = all_features(train_idx, :);
train_labels = all_labels(train_idx);
test_features = all_features(test_idx, :);
test_labels = all_labels(test_idx);

% Save the split data
save(fullfile(data_path, 'data.mat'), 'train_features', 'train_labels', ...
    'test_features', 'test_labels', 'image_size');

% Display information about the new split
fprintf('New training set size: %d samples\n', size(train_features, 1));
fprintf('New test set size: %d samples\n', size(test_features, 1));
fprintf('Class distribution in training set: %.2f%% class 1\n', ...
    100 * sum(train_labels) / length(train_labels));
fprintf('Class distribution in test set: %.2f%% class 1\n', ...
    100 * sum(test_labels) / length(test_labels));