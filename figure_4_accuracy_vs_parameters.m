% Load the dataset and results
load('data/SIN_2/data.mat');
load('accuracy_with_grid_search.mat');

% Create figure with adjusted size
close all; 
fig = figure;
pos = get(gcf,'Position');
scale = 0.78;
set(gcf,'Position',[pos(1),pos(2),pos(3)*scale*3,pos(4)*scale*2]);

% Define subplot layout parameters
Nh = 2; % Number of rows
Nw = 3; % Number of columns
gap = [0.15 0.06]; % [vertical, horizontal] gap
marg_h = [0.1 0.1]; % [lower, upper] margin
marg_w = [0.1 0.17]; % [left, right] margin

% Create tight subplot layout
[ha, pos] = tight_subplot(Nh, Nw, gap, marg_h, marg_w);

% 1. LR-L2
axes(ha(1));
plot(lg_lam_range, accuracies_L2, 'LineWidth', 2);
xlabel('lg(\lambda_2)');
ylabel('Accuracy');
title('(a) LR-L2');
grid on;

% 2. LR-L1
axes(ha(2));
plot(lg_lam_range, accuracies_L1, 'LineWidth', 2);
xlabel('lg(\lambda_1)');
ylabel('Accuracy');
title('(b) LR-L1');
grid on;

% 3. LR-ElasticNet
axes(ha(3));
imagesc(lg_lam_range, lg_lam_range, accuracies_EN);
xlabel('lg(\lambda_2)');
ylabel('lg(\lambda_1)');
title('(c) LR-ElasticNet');
axis xy;

% 4. LR-GraphNet
axes(ha(4));
imagesc(lg_lam_range, lg_lam_range, accuracies_GN);
xlabel('lg(\lambda_2)');
ylabel('lg(\lambda_1)');
title('(d) LR-GraphNet');
axis xy;

% 5. LR-SS1
axes(ha(5));
imagesc(lg_lam_range, lg_lam_range, accuracies_SS1);
xlabel('lg(\lambda_2)');
ylabel('lg(\lambda_1)');
title('(e) LR-SS1');
axis xy;

% 6. LR-SS2
axes(ha(6));
h = imagesc(lg_lam_range, lg_lam_range, accuracies_SS2);
xlabel('lg(\lambda_2)');
ylabel('lg(\lambda_1)');
title('(f) LR-SS2');
axis xy;

% Add colorbar
cb = colorbar('Position', [0.86 0.10 0.02 0.8]);  % Adjusted position

% Set figure properties
set(gcf, 'Color', 'white');

% Find and display maximum accuracies and corresponding parameters

% standard
% LR
paras = struct('lam1', 0, 'lam2', 0, 'opt', 1);
% Z-score normalization
[train_features, test_features] = improved_zscore(train_features, test_features);
[predict_labels, ~, ~] = SSLR(train_features, test_features, train_labels, paras);
accuracy = mean((predict_labels > 0.5) == test_labels);
fprintf('LR: Accuracy = %.4f\n', accuracy);

% L2 regularization
[max_acc_L2, idx_L2] = max(accuracies_L2);
lambda_L2 = 10^lg_lam_range(idx_L2);
fprintf('LR-L2: Max accuracy = %.4f, λ2 = %.4e (lg(λ2) = %.2f)\n', ...
    max_acc_L2, lambda_L2, lg_lam_range(idx_L2));

% L1 regularization
[max_acc_L1, idx_L1] = max(accuracies_L1);
lambda_L1 = 10^lg_lam_range(idx_L1);
fprintf('LR-L1: Max accuracy = %.4f, λ1 = %.4e (lg(λ1) = %.2f)\n', ...
    max_acc_L1, lambda_L1, lg_lam_range(idx_L1));

% ElasticNet
[max_acc_EN, idx_EN] = max(accuracies_EN(:));
[i_EN, j_EN] = ind2sub(size(accuracies_EN), idx_EN);
lambda1_EN = 10^lg_lam_range(i_EN);
lambda2_EN = 10^lg_lam_range(j_EN);
fprintf('LR-ElasticNet: Max accuracy = %.4f, λ1 = %.4e (lg(λ1) = %.2f), λ2 = %.4e (lg(λ2) = %.2f)\n', ...
    max_acc_EN, lambda1_EN, lg_lam_range(i_EN), lambda2_EN, lg_lam_range(j_EN));

% GraphNet
[max_acc_GN, idx_GN] = max(accuracies_GN(:));
[i_GN, j_GN] = ind2sub(size(accuracies_GN), idx_GN);
lambda1_GN = 10^lg_lam_range(i_GN);
lambda2_GN = 10^lg_lam_range(j_GN);
fprintf('LR-GraphNet: Max accuracy = %.4f, λ1 = %.4e (lg(λ1) = %.2f), λ2 = %.4e (lg(λ2) = %.2f)\n', ...
    max_acc_GN, lambda1_GN, lg_lam_range(i_GN), lambda2_GN, lg_lam_range(j_GN));

% SS1
[max_acc_SS1, idx_SS1] = max(accuracies_SS1(:));
[i_SS1, j_SS1] = ind2sub(size(accuracies_SS1), idx_SS1);
lambda1_SS1 = 10^lg_lam_range(i_SS1);
lambda2_SS1 = 10^lg_lam_range(j_SS1);
fprintf('LR-SS1: Max accuracy = %.4f, λ1 = %.4e (lg(λ1) = %.2f), λ2 = %.4e (lg(λ2) = %.2f)\n', ...
    max_acc_SS1, lambda1_SS1, lg_lam_range(i_SS1), lambda2_SS1, lg_lam_range(j_SS1));

% SS2
[max_acc_SS2, idx_SS2] = max(accuracies_SS2(:));
[i_SS2, j_SS2] = ind2sub(size(accuracies_SS2), idx_SS2);
lambda1_SS2 = 10^lg_lam_range(i_SS2);
lambda2_SS2 = 10^lg_lam_range(j_SS2);
fprintf('LR-SS2: Max accuracy = %.4f, λ1 = %.4e (lg(λ1) = %.2f), λ2 = %.4e (lg(λ2) = %.2f)\n', ...
    max_acc_SS2, lambda1_SS2, lg_lam_range(i_SS2), lambda2_SS2, lg_lam_range(j_SS2));

% export figure
folder = 'pdf';
output_filename = 'figure_4';
remkdir(folder);
exportgraphics(gcf, sprintf('%s/%s.pdf', folder, output_filename), 'ContentType', 'vector');
