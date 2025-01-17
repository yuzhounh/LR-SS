%% Data Preprocessing

% data path
fprintf('Loading DistalPhalanxOutlineCorrect...\n');
data_path = 'data/DistalPhalanxOutlineCorrect'; 
train_data_file = fullfile(data_path, 'DistalPhalanxOutlineCorrect_TRAIN.txt'); 
test_data_file = fullfile(data_path, 'DistalPhalanxOutlineCorrect_TEST.txt');

%% train 
% load data from txt file
train_data = load(train_data_file);

train_features = train_data(:,2:end);
train_labels = train_data(:,1);

%% test
% load data from txt file
test_data = load(test_data_file);

test_features = test_data(:,2:end);
test_labels = test_data(:,1);

%% Save processed data
save(fullfile(data_path, 'data.mat'), 'train_features', 'train_labels', 'test_features', 'test_labels');

%% Display dataset information
fprintf('Training set size: %d samples\n', size(train_features, 1));
fprintf('Test set size: %d samples\n', size(test_features, 1)); 
fprintf('Number of features per sample: %d\n', size(train_features, 2));
fprintf('\n');
