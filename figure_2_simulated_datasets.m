% Signal Visualization and Generation
% This script generates and visualizes three types of signals:
% 1. Sin wave
% 2. Random noise
% 3. Sin wave with added noise
% Created on: 2022-3-11
% Last modified: 2024-11-21

% Print status message
fprintf('Plotting samples for simulated datasets...\n\n');

% Configuration parameters
% rng(1);                     % Set random seed for reproducibility
n = 1000;                   % Number of samples to generate
d = 200;                    % Dimension of signals

% Signal ratio settings
sRatio = 2.^(0:4);          % Ratio array for signal scaling
iRatio = 2;                 % Selected ratio index
cRatio = 1 / sRatio(iRatio);% Scaling coefficient

% Generate sin wave signal
x = (1:200)';
x(1:80) = 0;
x(121:200) = 0;
sin_wave = sin(x * pi / 20) * cRatio;

% Generate noise signals
noise = randn(d, n);

% Generate noisy sin wave
noise_sin = randn(d, n) + repmat(sin_wave, [1, n]);

% Combine signals for consistent y-axis scaling
lines = [sin_wave; noise(:,1); noise_sin(:,1)];

% Create figure with three subplots
figure('Position', [100, 100, 800, 450]);

% Noise Subplot
subplot(3,1,1);
plot(noise(:,1), 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1.5);
ylim([min(lines(:)), max(lines(:))]);
title('(a) Class 0', 'FontWeight', 'bold');
% grid on;

% Noisy Sin Wave Subplot
subplot(3,1,2);
plot(noise_sin(:,1), 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1.5);
ylim([min(lines(:)), max(lines(:))]);
title('(b) Class 1', 'FontWeight', 'bold');
% grid on;

% Sin Wave Subplot
subplot(3,1,3);
plot(sin_wave, 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 1.5);
ylim([min(lines(:)), max(lines(:))]);
title('(c) Sinusoidal Signal', 'FontWeight', 'bold');
% grid on;

% % Adjust overall figure properties
% sgtitle('Signal Visualization', 'FontSize', 16, 'FontWeight', 'bold');
% set(gcf, 'Color', 'white');

% scale
pos=get(gcf,'Position');
scale=1.0;
set(gcf,'Position',[pos(1),pos(2),pos(3)*scale*1.,pos(4)*scale]);

% export figure
folder = 'pdf';
output_filename = 'figure_2';
remkdir(folder);
exportgraphics(gcf, sprintf('%s/%s.pdf', folder, output_filename), 'ContentType', 'vector');