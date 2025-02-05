%% Theoretical Analysis
% Compare smoothing matrices
figure_1_smooth_matrices;

%% Experimental Preparation
% Generate simulated data
generate_simulated_datasets;

% Plot samples from simulated datasets
figure_2_simulated_datasets;

% Load real-world datasets
load_DistalPhalanxOutlineCorrect;
load_FashionMNIST;
load_GunPoint;
load_MNIST;

% Resplit training and test sets for image datasets
split_FashionMNIST;
split_MNIST;

% Plot samples from real-world datasets and output Table 5
figure_3_real_world_datasets;

%% Experiment 1
% Find optimal parameters using grid search
grid_search;
grid_search_2;

% Plot classification accuracy versus parameters and output Table 6
figure_4_accuracy_vs_parameters;
figure_5_accuracy_vs_parameters;

% Draw weight vectors
figure_6_weight_vectors; 

%% Experiment 2
% Test on simulated datasets using Bayesian optimization
classify_simulated_datasets;

% Plot results and output Table 8
figure_7_accuracy_on_simulated_datasets;

%% Experiment 3
% Test on real-world datasets using Bayesian optimization
classify_time_series;
classify_images;

% Plot results and output Table 9
figure_8_accuracy_on_real_world_datasets;