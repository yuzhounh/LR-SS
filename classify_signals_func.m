function classify_signals_func(dataset, repetition)
% Bayesian Optimization

% Load data
load(sprintf('data/%s/data.mat', dataset));

% Modified validation split settings - 80-20 split, use repetition as random seed
rng(repetition);  % Set random seed for reproducibility
validation_ratio = 0.2;
[train_idx, val_idx] = crossvalind('HoldOut', train_labels, validation_ratio, 'Classes', unique(train_labels));

% Split data into train and validation sets
train_features_split = train_features(train_idx, :);
train_labels_split = train_labels(train_idx);
val_features = train_features(val_idx, :);
val_labels = train_labels(val_idx);

% Z-score normalization
[train_features_split, val_features, test_features] = improved_zscore_val(train_features_split, val_features, test_features);

% Merge training and validation data after normalization
train_features_merged = [train_features_split; val_features];
train_labels_merged = [train_labels_split; val_labels];

% Define Bayesian Optimization options
opts_lam1 = optimizableVariable('lam1', [1e-3, 1e3], 'Transform', 'log');
opts_lam2 = optimizableVariable('lam2', [1e-3, 1e3], 'Transform', 'log');
opts_delta = optimizableVariable('delta', [0.1, 2.0]);
opts_epsilon = optimizableVariable('epsilon', [1, 5], 'Type', 'integer');

% Define Bayesian Optimization options for different parameter counts
bayesopt_settings_1param = {'MaxObjectiveEvaluations', 50, 'UseParallel', true};
bayesopt_settings_2param = {'MaxObjectiveEvaluations', 100, 'UseParallel', true};
bayesopt_settings_4param = {'MaxObjectiveEvaluations', 200, 'UseParallel', true};

% Initialize results storage
methods = {'LR', 'LR-L2', 'LR-L1', 'LR-ElasticNet', 'LR-GraphNet', 'LR-SS1', 'LR-SS2'}';
best_accuracies = zeros(length(methods), 1);
best_parameters = cell(length(methods), 1);
best_weights = cell(length(methods), 1);
validation_accuracies = zeros(length(methods), 1);
best_precisions = zeros(length(methods), 1);
best_recalls = zeros(length(methods), 1);
best_f1_scores = zeros(length(methods), 1);
best_aucs = zeros(length(methods), 1);

% Loop through each method
for method_idx = 1:length(methods)
    fprintf('\nProcessing %s...\n', methods{method_idx});
    
    % Set default parameters
    paras = struct('lam1', 0, 'lam2', 0, 'opt', 1, 'delta', 1, 'epsilon', 3);
    
    % Bayesian Optimization for each method
    switch method_idx
        case 1 % LR (Standard Logistic Regression)
            paras.lam1 = 0;
            paras.lam2 = 0;
            paras.opt = 1;
            [predict_labels, ~, ~] = SSLR(train_features_split, val_features, train_labels_split, paras);
            validation_accuracies(method_idx) = mean((predict_labels > 0.5) == val_labels);
        
        case 2 % LR-L2 (1 parameter)
            paras.opt = 1;
            obj_fun = @(x)validation_objective_function(x, train_features_split, val_features, train_labels_split, val_labels, ...
                struct('lam1', 0, 'lam2', x.lam2, 'opt', paras.opt));
            results = bayesopt(obj_fun, opts_lam2, bayesopt_settings_1param{:});
            validation_accuracies(method_idx) = -results.MinObjective;
            paras.lam2 = results.XAtMinObjective.lam2;
            
        case 3 % LR-L1 (1 parameter)
            paras.opt = 1;
            obj_fun = @(x)validation_objective_function(x, train_features_split, val_features, train_labels_split, val_labels, ...
                struct('lam1', x.lam1, 'lam2', 0, 'opt', paras.opt));
            results = bayesopt(obj_fun, opts_lam1, bayesopt_settings_1param{:});
            validation_accuracies(method_idx) = -results.MinObjective;
            paras.lam1 = results.XAtMinObjective.lam1;
            
        case 4 % LR-ElasticNet (2 parameters)
            paras.opt = 1;
            obj_fun = @(x)validation_objective_function(x, train_features_split, val_features, train_labels_split, val_labels, ...
                struct('lam1', x.lam1, 'lam2', x.lam2, 'opt', paras.opt));
            results = bayesopt(obj_fun, [opts_lam1, opts_lam2], bayesopt_settings_2param{:});
            validation_accuracies(method_idx) = -results.MinObjective;
            paras.lam1 = results.XAtMinObjective.lam1;
            paras.lam2 = results.XAtMinObjective.lam2;
            
        case 5 % LR-GraphNet (2 parameters)
            paras.opt = 2;
            obj_fun = @(x)validation_objective_function(x, train_features_split, val_features, train_labels_split, val_labels, ...
                struct('lam1', x.lam1, 'lam2', x.lam2, 'opt', paras.opt));
            results = bayesopt(obj_fun, [opts_lam1, opts_lam2], bayesopt_settings_2param{:});
            validation_accuracies(method_idx) = -results.MinObjective;
            paras.lam1 = results.XAtMinObjective.lam1;
            paras.lam2 = results.XAtMinObjective.lam2;
            
        case 6 % LR-SS1 (4 parameters)
            paras.opt = 3;
            obj_fun = @(x)validation_objective_function(x, train_features_split, val_features, train_labels_split, val_labels, ...
                struct('lam1', x.lam1, 'lam2', x.lam2, 'opt', paras.opt, 'delta', x.delta, 'epsilon', x.epsilon));
            results = bayesopt(obj_fun, [opts_lam1, opts_lam2, opts_delta, opts_epsilon], bayesopt_settings_4param{:});
            validation_accuracies(method_idx) = -results.MinObjective;
            paras.lam1 = results.XAtMinObjective.lam1;
            paras.lam2 = results.XAtMinObjective.lam2;
            paras.delta = results.XAtMinObjective.delta;
            paras.epsilon = results.XAtMinObjective.epsilon;
            
        case 7 % LR-SS2 (4 parameters)
            paras.opt = 4;
            obj_fun = @(x)validation_objective_function(x, train_features_split, val_features, train_labels_split, val_labels, ...
                struct('lam1', x.lam1, 'lam2', x.lam2, 'opt', paras.opt, 'delta', x.delta, 'epsilon', x.epsilon));
            results = bayesopt(obj_fun, [opts_lam1, opts_lam2, opts_delta, opts_epsilon], bayesopt_settings_4param{:});
            validation_accuracies(method_idx) = -results.MinObjective;
            paras.lam1 = results.XAtMinObjective.lam1;
            paras.lam2 = results.XAtMinObjective.lam2;
            paras.delta = results.XAtMinObjective.delta;
            paras.epsilon = results.XAtMinObjective.epsilon;
    end

    % Use best parameters found in validation for final testing
    [predict_labels, weights, ~] = SSLR(train_features_merged, test_features, train_labels_merged, paras);
    predicted_classes = predict_labels > 0.5;
    best_accuracies(method_idx) = mean(predicted_classes == test_labels);
    best_parameters{method_idx} = paras;
    best_weights{method_idx} = weights;
    
    % Calculate additional metrics
    tp = sum(predicted_classes & test_labels);
    fp = sum(predicted_classes & ~test_labels);
    fn = sum(~predicted_classes & test_labels);
    
    best_precisions(method_idx) = tp / (tp + fp);
    best_recalls(method_idx) = tp / (tp + fn);
    best_f1_scores(method_idx) = 2 * (best_precisions(method_idx) * best_recalls(method_idx)) / ...
                                (best_precisions(method_idx) + best_recalls(method_idx));
    
    % Calculate AUC
    [~,~,~,best_aucs(method_idx)] = perfcurve(test_labels, predict_labels, 1);
    
    % Print results for this method
    fprintf('Validation accuracy: %.4f\n', validation_accuracies(method_idx));
    fprintf('Test accuracy: %.4f\n', best_accuracies(method_idx));
    fprintf('Precision: %.4f\n', best_precisions(method_idx));
    fprintf('Recall: %.4f\n', best_recalls(method_idx));
    fprintf('F1 Score: %.4f\n', best_f1_scores(method_idx));
    fprintf('AUC: %.4f\n', best_aucs(method_idx));
end

% Display final results
fprintf('\nFinal Classification Results:\n');
for i = 1:length(methods)
    fprintf('\n%s:\n', methods{i});
    fprintf('Best Test Accuracy: %.4f\n', best_accuracies(i));
    fprintf('Precision: %.4f\n', best_precisions(i));
    fprintf('Recall: %.4f\n', best_recalls(i));
    fprintf('F1 Score: %.4f\n', best_f1_scores(i));
    fprintf('AUC: %.4f\n', best_aucs(i));
    fprintf('Best Parameters:\n');
    disp(best_parameters{i});
end

% Save results to .mat file
save(sprintf('results/results_%s_rep%d.mat', dataset, repetition), 'methods', 'best_accuracies', 'best_parameters', ...
    'best_weights', 'validation_accuracies', 'best_precisions', 'best_recalls', ...
    'best_f1_scores', 'best_aucs');
close all;
end

% Modify objective function for validation
function accuracy = validation_objective_function(x, train_features, val_features, train_labels, val_labels, paras)
    if isfield(x, 'lam1')
        paras.lam1 = x.lam1;
    end
    if isfield(x, 'lam2')
        paras.lam2 = x.lam2;
    end
    if isfield(x, 'delta')
        paras.delta = x.delta;
    end
    if isfield(x, 'epsilon')
        paras.epsilon = x.epsilon;
    end
    
    [predict_labels, ~, ~] = SSLR(train_features, val_features, train_labels, paras);
    accuracy = mean((predict_labels > 0.5) == val_labels);
    accuracy = -accuracy;  % Negative because bayesopt minimizes
end