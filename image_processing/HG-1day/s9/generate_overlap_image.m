%%%Source    : Houston Methodist Research Institute
%%%Location  : Houston, TX.
%%%Origin    : November 1, 2018
%%%PI        : Mauro Ferrari
%%%Supervisor: Giulia Brachi
%%%Developer : Javier Ruiz Ramirez

function generate_overlap_image(filename) 
%This function takes as an input the full filename of a .tif file.
%This function expects the hydrogel plus tumor image with 3 channels, were the
%red channel corresponds to the hydrogel and the green channel corresponds to
%the tumor. The blue channel is not used.
%The output of this function is the overlap between the tumor and the hydrogel.
%Additionally, the quantification of the overlap is also given.

%probability_mask();
%return;

day_id   = 1;
slice_id = 9;

fid = fopen('boundary_data.txt');
boundary_data = textscan(fid, '%d32 %d32 %d32');
fclose(fid);

circle_data_file_names = {'circle_data.txt'};


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


    contrast_interval_in  = [20 50]/255.;
    [hydrogel_img, hydrogel_contrast, hydrogel_bw] = ...
        process_image_to_bw(...
        hydrogel_img, ...
        vertical_percent,...
        contrast_interval_in,...
        contrast_interval_out);
    
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

    contrast_interval_in  = [30 50]/255.;
    [tumor_img, tumor_contrast, tumor_bw] = ...
        process_image_to_bw_2(tumor_img,...
        vertical_percent,...
        contrast_interval_in,...
        contrast_interval_out);
    
    [row,col] = find(tumor_bw);
    
    tumor_bw = remove_circle(tumor_bw, circle_data_file_names);
    imshow(tumor_bw);
    plot_boundary(boundary_data);
    save_current_image(day_id, slice_id, 'tumor_bw');

    h = toc;
    disp(['Time to process Tumor: ', num2str(h)]);
 
end

%%%================================================================
%%%Overlap image

close all;

set(0, 'DefaultFigureVisible', 'on');

tic;
nonzero_pixels_of_hydrogel = nnz(hydrogel_bw);
nonzero_pixels_of_tumor    = nnz(tumor_bw);
overlap                    = nnz(hydrogel_bw .* tumor_bw);
overlap_percentage_wrt_tumor = overlap / nonzero_pixels_of_tumor * 100;
overlap_txt = sprintf('%d', fix(overlap_percentage_wrt_tumor));

fid = fopen('./overlap_information.csv', 'w');

fprintf(fid, 'Day                             , %d \n',day_id);
fprintf(fid, 'Slice                           , %d \n',...
    slice_id);
fprintf(fid, 'Nonzero pixels in hydrogel image, %d \n',...
    nonzero_pixels_of_hydrogel);
fprintf(fid, 'Nonzero pixels in tumor image   , %d \n',...
    nonzero_pixels_of_tumor);
fprintf(fid, 'Nonzero pixels in overlap image , %d \n',...
    overlap);
fprintf(fid, 'Overlap with respect to tumor   , %s%% \n',...
    overlap_txt);
fclose(fid);

t = ['Hydrogel + Tumor overlap: ', overlap_txt, '%'];

h = toc;
disp(['Time to compute overlap: ', num2str(h)]);

figure();
%horizontal_percent = 0.25;
horizontal_percent = 0.0;
starting_point     = max(uint32(width * horizontal_percent),1);
hydrogel_bw = hydrogel_bw(:, starting_point:end);
imshow(hydrogel_bw);
hold on;

sz = 2;
marker_color = 'b';
alpha_value  = 0.005;

tumor_bw = tumor_bw(:, starting_point:end);

[row_indices, col_indices] = find(tumor_bw);
h = scatter(col_indices, row_indices, sz,...
'MarkerFaceColor', marker_color, 'MarkerEdgeColor', marker_color,...
'MarkerFaceAlpha', alpha_value, 'MarkerEdgeAlpha', alpha_value);

%title(t);
%camroll(-90);

tfinal = toc;
disp(['Time to process image: ', num2str(tfinal)]);

plot_boundary(boundary_data);
save_current_image(day_id, slice_id, 'overlap')

%%%================================================================

function [img, img_contrast, img_bw] = ...
    process_image_to_bw(img,...
    vertical_percent,...
    contrast_interval_in,...
    contrast_interval_out)
%This function modifies the hydrogel image and generates 3 images.
%The first one is the croped original image.
%The second modifies the contrast of the first one.
%The third is a black and white version of the second.

s = size(img);
width = s(2);
height= s(1);

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
        contrast_interval_out)
% We use this function to modify the tumor image and generate two images.
%The first one is a croped version of the original.
%The second is a black and white version of the original.

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

function img = create_black_holes(img)
%Eliminate all pixels located inside a circle of radius r located at the
%point center.
%This function uses a CSV file that contains three columns (x,y,r).
%(x,y) denotes the center of a circle, and r its radius.

m = csvread('./centers_and_radii_for_elimination.csv');
center_vector = m(:,1:2);
radius_vector = m(:,3);

s = size(img);
n_rows = s(1);
n_cols = s(2);
[row_indices, col_indices] = find(img);
%Convert row and column indices to linear index
linear_indices = (col_indices-1) * n_rows + row_indices;

%Verify that linear indices were correctly computed
%true_linear_indices = find(img);
%if ~isequal(linear_indices, true_linear_indices)
    %error('Indices do not match');
%end

for k = 1:size(center_vector,1)

distance = sqrt((row_indices - center_vector(k,2)).^2 +...
    (col_indices - center_vector(k,1)).^2);
mask = radius_vector(k) < distance;

linear_indices = linear_indices(mask);
row_indices    = row_indices(mask);
col_indices    = col_indices(mask);

end

%Keep the pixels that are at a distance greater than that specified by each
%circle.

img = zeros(n_rows, n_cols, 'like', img);
img(linear_indices) = 1;



%%%================================================================
function plot_boundary(boundary_data)
hold on;
plot(boundary_data{2}, boundary_data{3}, 'b-', 'LineWidth', 3, 'LineStyle', '-');


%%%================================================================
function img = remove_circle(img, circle_data_file_names)

s = size(img);
n_rows = s(1);
n_cols = s(2);
[row_indices, col_indices] = find(img);
%Convert row and column indices to linear index
linear_indices = (col_indices-1) * n_rows + row_indices;

for k = 1:length(circle_data_file_names)

fname = circle_data_file_names{k};
fid   = fopen(fname);
circle_data = textscan(fid, '%d32 %d32 %d32');
x_data = double(circle_data{2});
y_data = double(circle_data{3});
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


mask     = rand(length(distance),1) < probability_mask(distance,r_opt);
%mask     = r_opt < distance; %For complement
%mask     = distance <= r_opt; %Interior

linear_indices = linear_indices(mask);
row_indices    = row_indices(mask);
col_indices    = col_indices(mask);


end

img = zeros(n_rows, n_cols, 'like', img);
img(linear_indices) = 1;

%tt = 0:pi/100:2*pi;
%xx = r_opt * cos(tt) + x_opt;
%yy = r_opt * sin(tt) + y_opt;
%
%
%hold on;
%plot(x_data, y_data, 'bo', 'LineStyle', 'None')
%plot(xx, yy, 'r-', 'LineWidth', 2)


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



