% Load the dataset and results
load('data/SIN_2/data.mat');

% Z-score normalization
[train_features, test_features] = improved_zscore(train_features, test_features);

% Parameter ranges for delta and epsilon
delta_range = 0.1:0.1:10;
epsilon_range = 1:1:10;

% Create figure
figure('Position', [100 100 800 400]);

% Calculate total iterations
total_iterations = length(delta_range) * length(epsilon_range) * 2; % For SS1 and SS2
current_iteration = 0;
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

% Fixed lambda values for SS1 and SS2
ss1_lam1 = 39.811; % 10^1.60
ss1_lam2 = 10000;  % 10^4.00
ss2_lam1 = 19.953; % 10^1.30
ss2_lam2 = 0.25119; % 10^-0.60

% 1. LR-SS1
fprintf('Processing LR-SS1...\n');
accuracies_SS1 = zeros(length(delta_range), length(epsilon_range));
for i = 1:length(delta_range)
    for j = 1:length(epsilon_range)
        paras = struct('lam1', ss1_lam1, 'lam2', ss1_lam2, 'opt', 3, ...
                      'delta', delta_range(i), 'epsilon', epsilon_range(j));
        [predict_labels, ~, ~] = SSLR(train_features, test_features, train_labels, paras);
        accuracies_SS1(i,j) = mean((predict_labels > 0.5) == test_labels);
        current_iteration = current_iteration + 1;
        update_progress(current_iteration, total_iterations, start_time);
    end
end
subplot(1,2,1);
imagesc(epsilon_range, delta_range, accuracies_SS1);
colorbar;
xlabel('epsilon');
ylabel('delta');
title('LR-SS1');
axis xy;

% 2. LR-SS2
fprintf('Processing LR-SS2...\n');
accuracies_SS2 = zeros(length(delta_range), length(epsilon_range));
for i = 1:length(delta_range)
    for j = 1:length(epsilon_range)
        paras = struct('lam1', ss2_lam1, 'lam2', ss2_lam2, 'opt', 4, ...
                      'delta', delta_range(i), 'epsilon', epsilon_range(j));
        [predict_labels, ~, ~] = SSLR(train_features, test_features, train_labels, paras);
        accuracies_SS2(i,j) = mean((predict_labels > 0.5) == test_labels);
        current_iteration = current_iteration + 1;
        update_progress(current_iteration, total_iterations, start_time);
    end
end
subplot(1,2,2);
imagesc(epsilon_range, delta_range, accuracies_SS2);
colorbar;
xlabel('epsilon');
ylabel('delta');
title('LR-SS2');
axis xy;

% Adjust spacing and title
sgtitle('Parameter Analysis (delta, epsilon) for SS1 and SS2 on SIN\_2 Dataset');
set(gcf, 'Color', 'white');

% Save results
save('accuracy_with_grid_search_2.mat', 'accuracies_SS1', 'accuracies_SS2', ...
     'delta_range', 'epsilon_range');

fprintf('All computations completed. Results saved to accuracy_with_grid_search_2.mat\n');
