% Load the dataset and results
load('data/SIN_2/data.mat');

% Z-score normalization (reuse from classify_signals_func)
[train_features, test_features] = improved_zscore(train_features, test_features);

% Parameter range
lam_range = 10.^(-6:0.1:6);
lg_lam_range = -6:0.1:6;

% Create figure
figure('Position', [100 100 1200 800]);

% Calculate total iterations for progress tracking
total_iterations = length(lam_range) * 2 + ... % For L1 and L2
                  length(lam_range)^2 * 4;     % For EN, GN, SS1, and SS2
current_iteration = 0;
last_percent = 0;
tic; % Start timing

% Start timing
start_time = tic;

% Add function for progress update
function update_progress(current, total, start_time)
    persistent last_percent
    if isempty(last_percent)
        last_percent = 0;
    end
    
    percent = floor(100 * current / total);
    if percent > last_percent
        elapsed_time = toc(start_time);
        estimated_total_time = elapsed_time / (current / total);
        remaining_time = estimated_total_time - elapsed_time;
        fprintf('Progress: %d%% (Est. remaining time: %.1f minutes)\n', ...
                percent, remaining_time/60);
        last_percent = percent;
    end
end

% 1. LR-L2
fprintf('Processing LR-L2...\n');
accuracies_L2 = zeros(size(lam_range));
for i = 1:length(lam_range)
    paras = struct('lam1', 0, 'lam2', lam_range(i), 'opt', 1);
    [predict_labels, ~, ~] = SSLR(train_features, test_features, train_labels, paras);
    accuracies_L2(i) = mean((predict_labels > 0.5) == test_labels);
    current_iteration = current_iteration + 1;
    update_progress(current_iteration, total_iterations, start_time);
end
subplot(2,3,1);
plot(lg_lam_range, accuracies_L2, 'LineWidth', 2);
xlabel('log_{10}(λ_2)');
ylabel('Accuracy');
title('LR-L2');
grid on;

% 2. LR-L1
fprintf('Processing LR-L1...\n');
accuracies_L1 = zeros(size(lam_range));
for i = 1:length(lam_range)
    paras = struct('lam1', lam_range(i), 'lam2', 0, 'opt', 1);
    [predict_labels, ~, ~] = SSLR(train_features, test_features, train_labels, paras);
    accuracies_L1(i) = mean((predict_labels > 0.5) == test_labels);
    current_iteration = current_iteration + 1;
    update_progress(current_iteration, total_iterations, start_time);
end
subplot(2,3,2);
plot(lg_lam_range, accuracies_L1, 'LineWidth', 2);
xlabel('log_{10}(λ_1)');
ylabel('Accuracy');
title('LR-L1');
grid on;

% 3. LR-ElasticNet
fprintf('Processing LR-ElasticNet...\n');
accuracies_EN = zeros(length(lam_range), length(lam_range));
for i = 1:length(lam_range)
    for j = 1:length(lam_range)
        paras = struct('lam1', lam_range(i), 'lam2', lam_range(j), 'opt', 1);
        [predict_labels, ~, ~] = SSLR(train_features, test_features, train_labels, paras);
        accuracies_EN(i,j) = mean((predict_labels > 0.5) == test_labels);
        current_iteration = current_iteration + 1;
        update_progress(current_iteration, total_iterations, start_time);
    end
end
subplot(2,3,3);
imagesc(lg_lam_range, lg_lam_range, accuracies_EN);
colorbar;
xlabel('log_{10}(λ_1)');
ylabel('log_{10}(λ_2)');
title('LR-ElasticNet');
axis xy;

% 4. LR-GraphNet
fprintf('Processing LR-GraphNet...\n');
accuracies_GN = zeros(length(lam_range), length(lam_range));
for i = 1:length(lam_range)
    for j = 1:length(lam_range)
        paras = struct('lam1', lam_range(i), 'lam2', lam_range(j), 'opt', 2);
        [predict_labels, ~, ~] = SSLR(train_features, test_features, train_labels, paras);
        accuracies_GN(i,j) = mean((predict_labels > 0.5) == test_labels);
        current_iteration = current_iteration + 1;
        update_progress(current_iteration, total_iterations, start_time);
    end
end
subplot(2,3,4);
imagesc(lg_lam_range, lg_lam_range, accuracies_GN);
colorbar;
xlabel('log_{10}(λ_1)');
ylabel('log_{10}(λ_2)');
title('LR-GraphNet');
axis xy;

% 5. LR-SS1
fprintf('Processing LR-SS1...\n');
accuracies_SS1 = zeros(length(lam_range), length(lam_range));
for i = 1:length(lam_range)
    for j = 1:length(lam_range)
        paras = struct('lam1', lam_range(i), 'lam2', lam_range(j), 'opt', 3, ...
                      'delta', 1, 'epsilon', 3);
        [predict_labels, ~, ~] = SSLR(train_features, test_features, train_labels, paras);
        accuracies_SS1(i,j) = mean((predict_labels > 0.5) == test_labels);
        current_iteration = current_iteration + 1;
        update_progress(current_iteration, total_iterations, start_time);
    end
end
subplot(2,3,5);
imagesc(lg_lam_range, lg_lam_range, accuracies_SS1);
colorbar;
xlabel('log_{10}(λ_1)');
ylabel('log_{10}(λ_2)');
title('LR-SS1');
axis xy;

% 6. LR-SS2
fprintf('Processing LR-SS2...\n');
accuracies_SS2 = zeros(length(lam_range), length(lam_range));
for i = 1:length(lam_range)
    for j = 1:length(lam_range)
        paras = struct('lam1', lam_range(i), 'lam2', lam_range(j), 'opt', 4, ...
                      'delta', 1, 'epsilon', 3);
        [predict_labels, ~, ~] = SSLR(train_features, test_features, train_labels, paras);
        accuracies_SS2(i,j) = mean((predict_labels > 0.5) == test_labels);
        current_iteration = current_iteration + 1;
        update_progress(current_iteration, total_iterations, start_time);
    end
end
subplot(2,3,6);
imagesc(lg_lam_range, lg_lam_range, accuracies_SS2);
colorbar;
xlabel('log_{10}(λ_1)');
ylabel('log_{10}(λ_2)');
title('LR-SS2');
axis xy;

% Adjust spacing
sgtitle('Parameter Analysis for Different Methods on SIN\_2 Dataset');
set(gcf, 'Color', 'white');

% Save results after all computations
save('accuracy_with_grid_search.mat', 'accuracies_L1', 'accuracies_L2', ...
     'accuracies_EN', 'accuracies_GN', 'accuracies_SS1', 'accuracies_SS2', ...
     'lam_range', 'lg_lam_range');

fprintf('All computations completed. Results saved to accuracy_with_grid_search.mat\n');
