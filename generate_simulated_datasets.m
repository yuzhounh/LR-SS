% 2022-3-11 18:18:20

clear,clc,close all;

% Change signal-to-noise ratios
sRatio = 2.^(0:3);  % 4 different ratios: 1, 2, 4, 8
nRatio = length(sRatio);

for iRatio = 1:nRatio
    cRatio = 1/sRatio(iRatio);
    mkdir(sprintf('data/SIN_%d', iRatio));
    
    n_samples = 1000;  % samples per class
    n_total = n_samples * 2;  % total samples
    signal_length = 200;
    
    % Generate sine signal (more clearly defined)
    t = (1:signal_length)';
    signal_sin = zeros(signal_length, 1);
    signal_sin(81:120) = sin(t(81:120)*pi/20);  % Only middle portion contains sine wave
    
    % Generate data for both classes
    noise_only = randn(n_samples, signal_length);  % Class 0: Pure noise
    noise_with_sin = randn(n_samples, signal_length) + ...
                     repmat(signal_sin', n_samples, 1) * cRatio;  % Class 1: Noise + Sine
    
    % Combine features and labels
    features = [noise_only; noise_with_sin];
    labels = [zeros(n_samples, 1); ones(n_samples, 1)];
    
    % Stratified sampling for train/test split using crossvalind
    [train_idx, test_idx] = crossvalind('HoldOut', labels, 0.5);  % 50% for training, 50% for testing
    
    % Split data
    train_features = features(train_idx, :);
    test_features = features(~train_idx, :);  % Note: test_idx is logical, use ~ for complement
    train_labels = labels(train_idx);
    test_labels = labels(~train_idx);
    
    % Save data
    save(sprintf('data/SIN_%d/data.mat', iRatio), ...
         'train_features', 'test_features', 'train_labels', 'test_labels');
end