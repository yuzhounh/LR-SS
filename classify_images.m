% Image Classification Script for Multiple Datasets
% clear, clc;

fprintf('Classifying image datasets using Bayesian optimization...\n');

% List of datasets to process
datasets = {
    'FashionMNIST';
    'MNIST'
};

% Outer loop for repetitions
for rep = 1:10
    fprintf('Running repetition %d/10\n', rep);
    
    % Process each dataset
    for i = 1:length(datasets)
        dataset = datasets{i};
        fprintf('Processing dataset: %s\n', dataset);
        
        % Call the classification function with current repetition number
        classify_images_func(dataset, rep);
        fprintf('Successfully processed %s\n', dataset);
    end
end