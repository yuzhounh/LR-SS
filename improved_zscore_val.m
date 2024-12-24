function [normalized_train, normalized_val, normalized_test] = improved_zscore_val(train_features, val_features, test_features)

% Training data normalization
[normalized_train, mu, sigma] = zscore(train_features);

% Features with zero standard deviation (includes all-zero features)
is_constant_feature = sigma == 0;

% Validation data normalization
normalized_val = (val_features - mu) ./ sigma;
normalized_val(:, is_constant_feature) = 0;

% Test data normalization
normalized_test = (test_features - mu) ./ sigma;
normalized_test(:, is_constant_feature) = 0;

end