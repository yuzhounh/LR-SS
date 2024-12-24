function [normalized_train, normalized_test] = improved_zscore(train_features, test_features)

% training data
[normalized_train, mu, sigma] = zscore(train_features); 

% test data
normalized_test = zeros(size(test_features));
for i = 1:size(train_features, 2)
    if all(train_features(:,i) == 0)
        normalized_test(:,i) = test_features(:,i);
    elseif sigma(i) == 0
        normalized_test(:,i) = test_features(:,i) - mu(i);
    else
        normalized_test(:,i) = (test_features(:,i) - mu(i)) / sigma(i);
    end
end