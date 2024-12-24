% ===================================
% Combined Visualization of Multiple Datasets
% ===================================

% Initialize figure
% ----------------
figure('Position', [100, 100, 1600, 600]);

% ===================================
% 1. DistalPhalanxOutlineCorrect Dataset
% ===================================
% Load data
load('data/DistalPhalanxOutlineCorrect/data.mat');

% Prepare data
class0_idx = find(train_labels == 0, 1);
class1_idx = find(train_labels == 1, 1);

% Visualize class 0
subplot(4,2,1);
plot(train_features(class0_idx,:), 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1.5);
title('(a) DistalPhalanxOutlineCorrect')
grid on;

% Visualize class 1
subplot(4,2,3);
plot(train_features(class1_idx,:), 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1.5);
grid on;


% ===================================
% 2. GunPoint Dataset
% ===================================
% Load data
load('data/GunPoint/data.mat');

% Prepare data
class0_idx = find(train_labels == 0, 1);
class1_idx = find(train_labels == 1, 1);

% Visualize class 0
subplot(4,2,2);
plot(train_features(class0_idx,:), 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1.5);
title('(b) GunPoint')
grid on;

% Visualize class 1
subplot(4,2,4);
plot(train_features(class1_idx,:), 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1.5);
grid on;


% ===================================
% 3. FashionMNIST Dataset
% ===================================
subplot(4,2,[5,7]);

% Load and prepare data
load('data/FashionMNIST/data.mat');
class0_idx = find(train_labels == 0);
class1_idx = find(train_labels == 1);

% Sample selection
num_samples = 6;
class0_samples = class0_idx(randperm(length(class0_idx), num_samples));
class1_samples = class1_idx(randperm(length(class1_idx), num_samples));
all_samples = [class0_samples; class1_samples];

% Image preparation
all_images = zeros(image_size(1), image_size(2), num_samples*2);
for i = 1:num_samples*2
    img = reshape(train_features(all_samples(i), :), image_size);
    all_images(:,:,i) = img;
end

% Visualization
montage(all_images, 'Size', [2, num_samples], 'DisplayRange', []);
title('(c) FashionMNIST');


% ===================================
% 4. MNIST Dataset
% ===================================
subplot(4,2,[6,8]);

% Load and prepare data
load('data/MNIST/data.mat');
class0_idx = find(train_labels == 0);
class1_idx = find(train_labels == 1);

% Sample selection
num_samples = 6;
class0_samples = class0_idx(randperm(length(class0_idx), num_samples));
class1_samples = class1_idx(randperm(length(class1_idx), num_samples));
all_samples = [class0_samples; class1_samples];

% Image preparation
all_images = zeros(image_size(1), image_size(2), num_samples*2);
for i = 1:num_samples*2
    img = reshape(train_features(all_samples(i), :), image_size);
    all_images(:,:,i) = img;
end

% Visualization
montage(all_images, 'Size', [2, num_samples], 'DisplayRange', []);
title('(d) MNIST');


% ===================================
% Export figure
% ===================================
folder = 'pdf';
output_filename = 'figure_3';
remkdir(folder);
exportgraphics(gcf, sprintf('%s/%s.pdf', folder, output_filename), 'ContentType', 'vector');