% Process and output classification results
clear, clc, close all;

% Define datasets and methods
datasets = {'DistalPhalanxOutlineCorrect','GunPoint', 'FashionMNIST', 'MNIST'};
methods = {'LR', 'LR-L2', 'LR-L1', 'LR-ElasticNet', 'LR-GraphNet', 'LR-SS1', 'LR-SS2'};
metrics = {'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'};

% Initialize cell array to store results for each dataset
all_results = cell(length(datasets), 1);

% Define number of repetitions for each dataset
NUM_REPETITIONS = zeros(length(datasets), 1);
NUM_REPETITIONS(1:2) = 30;  % First two datasets: 100 repetitions
NUM_REPETITIONS(3:4) = 10;   % Last two datasets: 10 repetitions

% Process each dataset
for d = 1:length(datasets)
    dataset = datasets{d};
    current_repetitions = NUM_REPETITIONS(d);
    
    % Initialize arrays to store metrics
    accuracies = zeros(length(methods), current_repetitions);
    precisions = zeros(length(methods), current_repetitions);
    recalls = zeros(length(methods), current_repetitions);
    f1_scores = zeros(length(methods), current_repetitions);
    aucs = zeros(length(methods), current_repetitions);
    
    % Load results from all repetitions
    for rep = 1:current_repetitions
        filename = sprintf('results/results_%s_rep%d.mat', dataset, rep);
        load(filename);
        
        accuracies(:, rep) = best_accuracies;
        precisions(:, rep) = best_precisions;
        recalls(:, rep) = best_recalls;
        f1_scores(:, rep) = best_f1_scores;
        aucs(:, rep) = best_aucs;
    end
    
    % Calculate means and standard deviations
    means = [mean(accuracies, 2), mean(precisions, 2), mean(recalls, 2), ...
            mean(f1_scores, 2), mean(aucs, 2)];
    stds = [std(accuracies, 0, 2), std(precisions, 0, 2), std(recalls, 0, 2), ...
           std(f1_scores, 0, 2), std(aucs, 0, 2)];
    
    % Store results for this dataset
    all_results{d} = struct('means', means, 'stds', stds);
    
    % Display results for this dataset
    fprintf('\nResults for %s:\n', dataset);
    fprintf('%-15s', 'Method');
    for m = 1:length(metrics)
        fprintf('%-25s', metrics{m});
    end
    fprintf('\n');
    fprintf('%s\n', repmat('-', 1, 140));
    
    for i = 1:length(methods)
        fprintf('%-15s', methods{i});
        for j = 1:size(means, 2)
            fprintf('%.3f ± %.3f          ', means(i,j), stds(i,j));
        end
        fprintf('\n');
    end
end

% Save all results
save('all_classification_results.mat', 'all_results', 'datasets', 'methods', 'metrics');

% After saving results, create bar plot for accuracy
% figure('Position', [100 100 1200 600]);
accuracy_means = zeros(length(methods), length(datasets));
accuracy_stds = zeros(length(methods), length(datasets));

% Extract accuracy data from all_results
for d = 1:length(datasets)
    accuracy_means(:,d) = all_results{d}.means(:,1);  % Column 1 is accuracy
    accuracy_stds(:,d) = all_results{d}.stds(:,1);    % Column 1 is accuracy
end

% Transpose matrices to group by dataset
accuracy_means = accuracy_means';
accuracy_stds = accuracy_stds';

% Create grouped bar plot
b = bar(accuracy_means);
hold on;

% Add error bars
[ngroups, nbars] = size(accuracy_means);
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
    errorbar(x(i,:), accuracy_means(:,i), accuracy_stds(:,i), 'k', 'LineStyle', 'none');
end

% Customize plot
% title('Classification Accuracy Across Datasets', 'FontSize', 14);
xlabel('Datasets', 'FontSize', 12);
ylabel('Accuracy', 'FontSize', 12);
datasets = {'DistalPhalanxOutlineCorrect','GunPoint', 'FashionMNIST', 'MNIST'};
xticklabels(datasets);
xtickangle(0);
legend(methods, 'Location', 'northwest');
grid on;

% Adjust figure
set(gca, 'FontSize', 10);
ylim([0.35 1.05]);  % Adjusted to potentially accommodate lower accuracy values

% Scale
pos=get(gcf,'Position');
scale=1.6;
set(gcf,'Position',[pos(1),pos(2),pos(3)*scale,pos(4)*scale*0.8]);

% export figure
folder = 'pdf';
output_filename = 'figure_8';
remkdir(folder);
exportgraphics(gcf, sprintf('%s/%s.pdf', folder, output_filename), 'ContentType', 'vector');