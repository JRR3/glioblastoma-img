%%%Source    : Houston Methodist Research Institute
%%%Location  : Houston, TX.
%%%Origin    : November 1, 2018
%%%PI        : Mauro Ferrari
%%%Supervisor: Giulia Brachi
%%%Developer : Javier Ruiz Ramirez

function generate_colorbar_image(filename) 
%This function takes as an input the full filename of a .tif file.
%This function expects the hydrogel plus tumor image with 3 channels, were the
%red channel corresponds to the hydrogel and the green channel corresponds to
%the tumor. The blue channel is not used.
%The output of this function is the overlap between the tumor and the hydrogel.
%Additionally, the quantification of the overlap is also given.

%probability_mask();
%return;

day_id   = 1;
slice_id = 6;


fid = fopen('boundary_data.txt');
boundary_data = textscan(fid, '%f %f %f');
fclose(fid);

tumor_circle_data_file_names   {1} = 'circle_data_tumor_1.txt';
tumor_circle_data_file_names   {2} = 'circle_data_tumor_2.txt';
hydrogel_circle_data_file_names{1} = 'circle_data_hydrogel_1.txt';


close all;
clc;
if nargin == 0
    hydrogel_plus_tumor  = imread('hydro_gel_plus_tumor.tif');
else
    hydrogel_plus_tumor  = imread(filename);
end

set(0, 'DefaultFigureVisible', 'on');

plot_hydrogel = true;
plot_tumor    = true;

%Load file

%hydrogel_plus_tumor  = imread('hydro_gel_plus_tumor.tif');
s = size(hydrogel_plus_tumor);
disp(['Original size: ', mat2str(s)]);


%Note the inversion. The first component of the matrix represents the
%vertical directon while the second component represents the horizontal
%direction.
width = s(2);
height= s(1);

%This is the red channel and corresponds to the hydrogel
hydrogel_img = hydrogel_plus_tumor(:,:,1);

%This is the green channel and corresponds to the tumor
tumor_img    = hydrogel_plus_tumor(:,:,2);


%Free memory
hydrogel_plus_tumor = [];

vertical_percent    = 0.0;

contrast_interval_out = [0 255]/255.;

  

%--------------------------------------------Plot hydrogel 
if plot_hydrogel    
    tic;
    close all;



    contrast_interval_in  = [35 50]/255.;
    [hydrogel_img, hydrogel_contrast, hydrogel_bw] = ...
        process_image_to_bw(...
        hydrogel_img, ...
        vertical_percent,...
        contrast_interval_in,...
        contrast_interval_out,... 
        hydrogel_circle_data_file_names);
    
    %imshow(hydrogel_bw);
    imshow(hydrogel_bw);
    plot_boundary(boundary_data);

    save_current_image(day_id, slice_id, 'hydrogel_bw');

    h = toc;
    disp(['Time to process hydrogel: ', num2str(h)]);
end

%--------------------------------------------Plot tumor
if plot_tumor
    tic;
    close all;

    contrast_interval_in  = [15 50]/255.;
    [tumor_img, tumor_contrast, tumor_bw] = ...
        process_image_to_bw_2(tumor_img,...
        vertical_percent,...
        contrast_interval_in,...
        contrast_interval_out,...
        tumor_circle_data_file_names);
    
    imshow(tumor_bw);
    plot_boundary(boundary_data);
    save_current_image(day_id, slice_id, 'tumor_bw');

    h = toc;
    disp(['Time to process Tumor: ', num2str(h)]);
 
end


%%%================================================================
%In this section we plot the combination of the hydrogel gray image using 
%a colorbar.

close all;
set(0, 'DefaultFigureVisible', 'on');
tic;

figure();

%Crop figure from left
s = size(hydrogel_contrast);
width = s(2);
height= s(1);

%Crop images
horizontal_percent = 0.0;
starting_point     = max(uint32(width * horizontal_percent),1);
hydrogel_contrast = hydrogel_contrast(:, starting_point:end);
imshow(hydrogel_contrast);

%Colormap for hydrogel
mx   = max(hydrogel_contrast(:));
cmap = jet(255);
rg = 1;
cmap(1:rg,:) = 0;
colormap(cmap);
cb = colorbar();
set(cb,'location', 'west');
set(cb,'Color', 'w');
hold on;
plot_boundary(boundary_data);

sz = 2;
marker_color = 'm';
alpha_value  = 0.004;

tumor_bw = tumor_bw(:, starting_point:end);
[row_indices, col_indices] = find(tumor_bw);

%mask = starting_point < col_indices;
%col_indices = col_indices(mask);
%row_indices = row_indices(mask);

h = scatter(col_indices, row_indices, sz,...
'MarkerFaceColor', marker_color, 'MarkerEdgeColor', marker_color,...
'MarkerFaceAlpha', alpha_value, 'MarkerEdgeAlpha', alpha_value);

set(gca, 'FontSize', 16);

%Rotate camera 90 degrees clockwise
%camroll(-90);


tfinal = toc;
disp(['Time to process image: ', num2str(tfinal)]);

save_current_image(day_id, slice_id, 'colorbar')

%%%================================================================

function [img, img_contrast, img_bw] = ...
    process_image_to_bw(img,...
    vertical_percent,...
    contrast_interval_in,...
    contrast_interval_out,...
    circle_data_file_names)
%This function modifies the hydrogel image and generates 3 images.
%The first one is the croped original image.
%The second modifies the contrast of the first one.
%The third is a black and white version of the second.

s = size(img);
width = s(2);
height= s(1);

img = remove_circle(img, circle_data_file_names);

%Crop images
starting_point = max(uint32(height * vertical_percent),1);
img            = img(starting_point:end,:);
img_contrast   = [];
img_bw         = [];

%Map intensity values (continuously) to new values in the interval [x, y]
img_contrast = imadjust(img, contrast_interval_in, contrast_interval_out);

%Convert gray scale to black and white using Otsu's method
%Default sensitivity is set to 0.50
img_bw = imbinarize(img_contrast);


%%%================================================================

function [img, img_contrast, img_bw] = ...
    process_image_to_bw_2(img,...
        vertical_percent,...
        contrast_interval_in,...
        contrast_interval_out,...
        circle_data_file_names)
% We use this function to modify the tumor image and generate two images.
%The first one is a croped version of the original.
%The second is a black and white version of the original.

img = remove_circle(img, circle_data_file_names);

s = size(img);

width = s(2);
height= s(1);

img_contrast = [];
img_bw       = [];
threshold    = 0;

%Crop images
starting_point = max(uint32(height * vertical_percent),1);
img            = img(starting_point:end, :);
img_contrast   = imadjust(img, contrast_interval_in, contrast_interval_out);
%img_bw         = imbinarize(img_contrast);  
img_bw         = threshold < img_contrast;
%img_bw         = create_black_holes(img_bw);



%%%================================================================

function save_current_image(day_id, slice_id, fname)
txt = [fname, '_day_', num2str(day_id), '_slice_', num2str(slice_id)];
print(txt, '-djpeg');



%%%================================================================
function plot_boundary(boundary_data)

hold on;

x = boundary_data{2};
y = boundary_data{3};

plot(x, y, 'b-', 'LineWidth', 3, 'LineStyle', '-');

%for k = 1:length(x)
    %if mod(k,1000) == 0
    %text(x(k), y(k), num2str(k), 'FontSize', 12, 'Color', 'w');
    %end
%end


%%%================================================================
function img = remove_circle(img, circle_data_file_names)

%plot_circles = true;
plot_circles = false;

s = size(img);
n_rows = s(1);
n_cols = s(2);
[row_indices, col_indices] = find(img);
%Convert row and column indices to linear index
linear_indices = (col_indices-1) * n_rows + row_indices;
main_mask      = false(length(linear_indices),1);

for k = 1:length(circle_data_file_names)

fname = circle_data_file_names{k};
fid   = fopen(fname);
circle_data = textscan(fid, '%f %f %f');
x_data = circle_data{2};
y_data = circle_data{3};
fclose(fid);

obj_fun = @(p) p(3).^2 - (x_data - p(1)).^2 - (y_data - p(2)).^2;
x_mean = mean(x_data);
y_mean = mean(y_data);
r_mean = norm([x_mean, y_mean] - [x_data(1), y_data(1)]);
opt_p  = lsqnonlin(obj_fun, [x_mean, y_mean, r_mean]);

x_opt  = opt_p(1);
y_opt  = opt_p(2);
r_opt  = opt_p(3);


distance = sqrt((row_indices - y_opt).^2 + (col_indices - x_opt).^2);


mask      = rand(length(distance),1) < probability_mask(distance, r_opt);
main_mask = main_mask | mask;

%mask     = r_opt < distance; %For complement
%mask     = distance <= r_opt; %Interior

%Remove comments to allow overwriting the mask
%linear_indices = linear_indices(mask);
%row_indices    = row_indices(mask);
%col_indices    = col_indices(mask);


if plot_circles
    tt = 0:pi/100:2*pi;
    xx = r_opt * cos(tt) + x_opt;
    yy = r_opt * sin(tt) + y_opt;
    hold on;
    plot(x_data, y_data, 'bo', 'LineStyle', 'None')
    plot(xx, yy, 'r-', 'LineWidth', 2)
end

end

linear_indices = linear_indices(~main_mask);
img(linear_indices) = 0;





%%%================================================================
function value = probability_mask(r, R)

if nargin == 0
R = 1;
r = 0:0.1:10;
else
r = r / R;
end

value      = r <= 1;
complement = 1 - value;
k          = 1;
constant   = exp(k * 1);
probability = constant * exp(-k * r);
value       = value + complement .* probability;
low_pass    = r <= 1.3;
value       = value .* low_pass;

if nargin == 0
figure();
plot(r, value, 'b-', 'LineWidth', 2);
end



