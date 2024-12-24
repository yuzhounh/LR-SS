function [label_predict, w, iter] = SSLR(x_train, x_test, label_train, paras)
% Optimized Smooth Sparse Logistic Regression (SSLR) with constant feature
% Input:
%   x_train: training features
%   x_test: testing features
%   label_train: training labels
%   paras: parameter struct with lam1, lam2, and smooth_matrix function

% Extract parameters
lam1 = paras.lam1;
lam2 = paras.lam2;

% Transpose and prepare data
x_train = x_train';
x_test = x_test';

% Add constant feature (bias term) to training and test data
x_train = [ones(1, size(x_train, 2)); x_train];
x_test = [ones(1, size(x_test, 2)); x_test];

% Get dimensions
[d, ~] = size(x_train); 

% Calculate the distances between features
distance=squareform(pdist([1:d-1]'));

% Compute initial weights
a = -sum(x_train.^2, 2) / 4;
w = ones(d, 1);

% Optimization parameters
max_iter = 1000;
tolerance = 1e-3;

% Iterative optimization
for iter = 1:max_iter
    % Store previous weights
    w_prev = w;

    % Compute sigmoid
    s = 1 ./ (1 + exp(-x_train' * w));

    % Gradient computation
    g = x_train * (label_train - s);

    if paras.opt == 1 % LR, LR-L1, LR-L2, LR-ElasticNet
        w = soft(-a .* w + g, lam1) ./ (-a + lam2 * ones(d,1) + eps);
    elseif paras.opt == 2 || paras.opt == 3 % LR-GraphNet, LR-SS1
        N = adjacency_matrix(distance, paras); % Compute for non-constant features
        N = blkdiag(0, N);                  % Extend Q with 0 for constant feature
        w = soft(-a .* w + lam2 * N * w + g, lam1) ./ (-a + lam2 * N' * ones(d,1) + eps);
    elseif paras.opt == 4 % LR-SS2
        Q = smooth_matrix(distance, paras); % Compute for non-constant features
        Q = blkdiag(0, Q);              % Extend Q with 0 for constant feature
        q = diag(Q);
        w = soft((-a + lam2 * q) .* w - lam2 * Q * w + g, lam1) ./ (-a + lam2 * q + eps);
    end

    % Check convergence
    err = norm(w - w_prev) / norm(w_prev);
    if err <= tolerance
        break;
    end
end

% Predict test labels
label_predict = 1 ./ (1 + exp(-x_test' * w));

% remove the intercept term from the returned weights
w(1) = [];

end

% Soft thresholding operator
function y = soft(x, lambda)
y = sign(x) .* max(abs(x) - lambda, 0);
end

% Adjacency matrix
function N = adjacency_matrix(distance, paras)
if paras.opt == 2 % GraphNet
    d = size(distance, 1); 
    N = ones(d,d).*(distance == 1);
elseif paras.opt ==3 % LR-SS1
    delta = paras.delta;
    epsilon = paras.epsilon;
    N = exp(-distance.^2/delta^2/2).*(0 < distance & distance <= epsilon); 
end
end

% Smooth matrix for LR-SS2
function Q = smooth_matrix(distance, paras)
delta = paras.delta;
epsilon = paras.epsilon;
N = exp(-distance.^2/delta^2/2).*(0 < distance & distance <= epsilon); 
Q = pinv(N);
end