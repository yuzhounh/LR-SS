% Signal Classification Script for Multiple Datasets
% clear, clc;

% List of datasets to process
datasets = {
    'SIN_1';
    'SIN_2';
    'SIN_3';
    'SIN_4'
};

% Outer loop for repetitions
for rep = 1:100
    fprintf('Running repetition %d/100...\n', rep);
    
    % Process each dataset
    for i = 1:length(datasets)
        dataset = datasets{i};
        fprintf('Processing dataset: %s\n', dataset);
        classify_signals_func(dataset, rep);
        fprintf('Successfully processed %s\n', dataset);
    end
end
