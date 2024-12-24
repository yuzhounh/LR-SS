% Signal Classification Script for Multiple Datasets
clear, clc;

% List of datasets to process
datasets = {
    'DistalPhalanxOutlineCorrect';
    'GunPoint'
};

% Outer loop for repetitions
for rep = 1:30
    fprintf('Running repetition %d/30...\n', rep);
    
    % Process each dataset
    for i = 1:length(datasets)
        dataset = datasets{i};
        fprintf('Processing dataset: %s\n', dataset);
        classify_signals_func(dataset, rep);
        fprintf('Successfully processed %s\n', dataset);
    end
end
