% close all; 
figure; 

% load results
load('accuracy_with_grid_search_2.mat', 'accuracies_SS1', 'accuracies_SS2', ...
     'delta_range', 'epsilon_range');

% Define subplot layout parameters
Nh = 1; % Number of rows
Nw = 2; % Number of columns
gap = [0.4 0.08]; % [vertical, horizontal] gap
marg_h = [0.15 0.1]; % [lower, upper] margin
marg_w = [0.1 0.1]; % [left, right] margin
[ha, pos] = tight_subplot(Nh, Nw, gap, marg_h, marg_w); % Create tight subplot layout

% scale
pos=get(gcf,'Position');
scale=0.7;
set(gcf,'Position',[pos(1),pos(2),pos(3)*scale*2.1,pos(4)*scale]);

% First subplot
axes(ha(1));
imagesc(epsilon_range, delta_range, accuracies_SS1);
colorbar;
xlabel('$\varepsilon$','Interpreter','latex');
ylabel('$\delta$','Interpreter','latex');
title('(a) LR-SS1');
axis xy;

% Second subplot
axes(ha(2));
imagesc(epsilon_range, delta_range, accuracies_SS2);
colorbar;
xlabel('$\varepsilon$','Interpreter','latex');
ylabel('$\delta$','Interpreter','latex');
title('(b) LR-SS2');
axis xy;

% Adjust spacing and title
set(gcf, 'Color', 'white');

% export figure
folder = 'pdf';
output_filename = 'figure_5';
remkdir(folder);
exportgraphics(gcf, sprintf('%s/%s.pdf', folder, output_filename), 'ContentType', 'vector');