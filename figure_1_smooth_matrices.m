% 2022-7-5 13:52:29
clear,clc,close all;
warning('off');

d=11;
epsilon=3; % 1, 2, 3, 10, Inf

% Create figure
fig = figure;
pos = get(gcf,'Position');
scale = 0.7;
set(gcf,'Position',[pos(1),pos(2),pos(3)*scale*3,pos(4)*scale*2]);

% Define the number of rows and columns
Nh = 2; % Number of rows
Nw = 3; % Number of columns

% Define the gaps and margins
gap = [0.11 0.04]; % [vertical, horizontal] gap
marg_h = [0.1 0.1]; % [lower, upper] margin
marg_w = [0.1 0.17]; % [left, right] margin

% Create the tight subplot layout
[ha, pos] = tight_subplot(Nh, Nw, gap, marg_h, marg_w);

% Encapsulate matrix generation logic into a function
function Q = generateMatrix(type, d, epsilon, delta)
    distance = squareform(pdist([1:d]'));
    
    switch type
        case 'eye'
            Q = eye(d);
        case 'Q1'
            A = exp(-distance.^2/delta^2/2).*(0 < distance & distance <= epsilon);
            D = diag(sum(A, 2));  % Use sum instead of ones(1,d)*A
            Q = D - A;
        case 'graphnet'
            A = ones(d,d).*(0 < distance & distance == 1);
            D = diag(sum(A, 2));
            Q = D - A;
        case 'Q2'
            A = exp(-distance.^2/delta^2/2).*(0 < distance & distance <= epsilon);
            Q = inv(A);
    end
end

% Main loop optimization
matrices = {
    struct('type', 'eye',      'delta', 0,   'title', '(a) Identity Matrix'),
    struct('type', 'Q1',       'delta', 0.8, 'title', '(b) Smooth Matrix $Q^{(1)}$ ($\delta=0.8$, $\varepsilon=3$)'),
    struct('type', 'Q1',       'delta', 1.6, 'title', '(c) Smooth Matrix $Q^{(1)}$ ($\delta=1.6$, $\varepsilon=3$)'),
    struct('type', 'graphnet', 'delta', 0,   'title', '(d) Smooth Matrix $Q^{(1)}$ ($\varepsilon=1$)'),
    struct('type', 'Q2',       'delta', 0.8, 'title', '(e) Smooth Matrix $Q^{(2)}$ ($\delta=0.8$, $\varepsilon=3$)'),
    struct('type', 'Q2',       'delta', 1.6, 'title', '(f) Smooth Matrix $Q^{(2)}$ ($\delta=1.6$, $\varepsilon=3$)')
};

for i = 1:6
    Q = generateMatrix(matrices{i}.type, d, epsilon, matrices{i}.delta);
    
    axes(ha(i));
    imagesc(Q);
    colorbar;
    title(['\sffamily \boldmath ', matrices{i}.title], 'FontSize', 10, 'Interpreter', 'latex');

    fprintf('Matrix %s:\n', matrices{i}.title);
    disp(Q);
    fprintf('\n');
end

% export figure
folder = 'pdf';
output_filename = 'figure_1';
remkdir(folder);
exportgraphics(gcf, sprintf('%s/%s.pdf', folder, output_filename), 'ContentType', 'vector');