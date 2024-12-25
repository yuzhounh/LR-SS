% Load the dataset
load('../data/SIN_2/data.mat');

% Z-score normalization
[train_features, test_features] = improved_zscore(train_features, test_features);

% Initialize cell arrays to store weights
methods = {'LR', 'LR-L2', 'LR-L1', 'LR-ElasticNet', 'LR-GraphNet', 'LR-SS1', 'LR-SS2'};
best_weights = cell(length(methods), 1);

% Train each method with optimal parameters and store weights
% 0. LR
paras = struct('lam1', 0, 'lam2', 0, 'opt', 1);
[~, weights, ~] = SSLR(train_features, test_features, train_labels, paras);
best_weights{1} = weights;

% 1. LR-L2
lambda2 = 10^4.90;
paras = struct('lam1', 0, 'lam2', lambda2, 'opt', 1);
[~, weights, ~] = SSLR(train_features, test_features, train_labels, paras);
best_weights{2} = weights;

% 2. LR-L1
lambda1 = 10^1.30;
paras = struct('lam1', lambda1, 'lam2', 0, 'opt', 1);
[~, weights, ~] = SSLR(train_features, test_features, train_labels, paras);
best_weights{3} = weights;

% 3. LR-ElasticNet
lambda1 = 10^1.30; lambda2 = 10^2.10;
paras = struct('lam1', lambda1, 'lam2', lambda2, 'opt', 1);
[~, weights, ~] = SSLR(train_features, test_features, train_labels, paras);
best_weights{4} = weights;

% 4. LR-GraphNet
lambda1 = 10^0.70; lambda2 = 10^2.70;
paras = struct('lam1', lambda1, 'lam2', lambda2, 'opt', 2);
[~, weights, ~] = SSLR(train_features, test_features, train_labels, paras);
best_weights{5} = weights;

% 5. LR-SS1
lambda1 = 10^1.60; lambda2 = 10^4.00;
paras = struct('lam1', lambda1, 'lam2', lambda2, 'opt', 3, 'delta', 1, 'epsilon', 3);
[~, weights, ~] = SSLR(train_features, test_features, train_labels, paras);
best_weights{6} = weights;

% 6. LR-SS2
lambda1 = 10^1.30; lambda2 = 10^(-0.60);
paras = struct('lam1', lambda1, 'lam2', lambda2, 'opt', 4, 'delta', 1, 'epsilon', 3);
[~, weights, ~] = SSLR(train_features, test_features, train_labels, paras);
best_weights{7} = weights;

% Save the weights
save('weight_vectors.mat', 'methods', 'best_weights');

% Load the weights file
load('weight_vectors.mat');

% Initialize arrays to store metrics
sparsity = zeros(length(methods), 1);
smoothness = zeros(length(methods), 1);

% Calculate sparsity and smoothness for each method
for i = 1:length(methods)
    % Calculate sparsity (percentage of zero elements)
    % Using a small threshold to account for numerical precision
    threshold = 1e-10;
    sparsity(i) = 100 * sum(abs(best_weights{i}) < threshold) / length(best_weights{i});
    
    % Calculate smoothness (sum of squared differences between adjacent elements)
    diff_vector = diff(best_weights{i});
    smoothness(i) = sum(diff_vector.^2);
end

% Display results
fprintf('\nSparsity and Smoothness Statistics:\n');
fprintf('----------------------------------------\n');
fprintf('Method\t\tSparsity(%%)\tSmoothness\n');
fprintf('----------------------------------------\n');
for i = 1:length(methods)
    fprintf('%s\t\t%.2f\t\t%.2e\n', methods{i}, sparsity(i), smoothness(i));
end
fprintf('----------------------------------------\n\n');

% Create a figure with 7x1 subplots
close all; 
figure('Position', [100, 100, 800, 1200]);  % Adjust figure size for better visibility

% Define colors
colors = [
    0, 0.4470, 0.7410;      % Blue
    0.8500, 0.3250, 0.0980; % Orange
    0.9290, 0.6940, 0.1250; % Yellow
    0.4940, 0.1840, 0.5560; % Purple
    0.4660, 0.6740, 0.1880; % Green
    0.3010, 0.7450, 0.9330; % Cyan
    0.6350, 0.0780, 0.1840  % Red
];

% Loop through each method
for i = 1:length(methods)
    % Create subplot
    subplot(7, 1, i);
    
    % Plot weights with corresponding color
    plot(best_weights{i}, 'Color', colors(i,:), 'LineWidth', 1.5);
    
    % Add title with letter label
    title([sprintf('(%s) ', char('a' + i - 1)), methods{i}], 'Interpreter', 'none');
    
    % % Add grid
    % grid on;
end

% export figure
folder = 'pdf';
output_filename = 'figure_6';
remkdir(folder);
exportgraphics(gcf, sprintf('%s/%s.pdf', folder, output_filename), 'ContentType', 'vector');