function find_clusters()
clc;
close all;
rng default; % For reproducibility
X = [randn(100,2)*0.75+ones(100,2);
    randn(100,2)*0.5-ones(100,2)];

disp('Size of X')
size(X)

figure;
% plot(X(:,1),X(:,2),'.');
x = -2*pi:0.05:+2*pi;
subplot(1,2,1);
hold on;
plot(x, sin(x));
% plot point at (x=0, y=0)
plot(0, sin(0), 'bx', 'ButtonDownFcn', {@getPoint);
% opts = statset('Display','final');

disp('Size of idx')
size(idx)

function getPoint(varargin)
currentPoint = get(gca, 'CurrentPoint');
fprintf('Hit Point! Coordinates: %f, %f \n', ...
    currentPoint(1), currentPoint(3));