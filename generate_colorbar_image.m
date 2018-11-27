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




close all;
clc;

set(0, 'DefaultFigureVisible', 'off');

plot_hydrogel = true;
plot_tumor    = true;

%Load file
hydrogel_plus_tumor  = imread(filename);
%hydrogel_plus_tumor  = imread('hydro_gel_plus_tumor.tif');
s                    = size(hydrogel_plus_tumor);
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

vertical_percent  = 0.40;

contrast_interval_in  = [20 50]/255.;
contrast_interval_out = [0 255]/255.;

  

%--------------------------------------------Plot hydrogel 
if plot_hydrogel    
    tic;

    [hydrogel_img, hydrogel_contrast, hydrogel_bw] = ...
        process_image_to_bw(...
        hydrogel_img, ...
        vertical_percent,...
        contrast_interval_in,...
        contrast_interval_out);

    h = toc;
    disp(['Time to process hydrogel: ', num2str(h)]);
end
%--------------------------------------------Plot tumor
if plot_tumor
    tic;

    [tumor_img, tumor_contrast, tumor_bw] = ...
        process_image_to_bw_2(tumor_img,...
        vertical_percent,...
        contrast_interval_in,...
        contrast_interval_out);

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
horizontal_percent = 0.25;
starting_point     = uint32(width * horizontal_percent);
hydrogel_contrast = hydrogel_contrast(:, starting_point:end);
imshow(hydrogel_contrast);

%Colormap for hydrogel
mx   = max(hydrogel_contrast(:));
cmap = jet(255);
rg = 1;
cmap(1:rg,:) = 0;
colormap(cmap);
cb = colorbar();
set(cb,'location', 'south');
set(cb,'Color', 'w');
hold on;

sz = 2;
marker_color = 'm';
alpha_value  = 0.008;

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
camroll(-90);


tfinal = toc;
disp(['Time to process image: ', num2str(tfinal)]);


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
starting_point = height * vertical_percent;
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
starting_point = height * vertical_percent;
img            = img(starting_point:end, :);
img_contrast   = imadjust(img, contrast_interval_in, contrast_interval_out);
%img_bw         = imbinarize(img_contrast);  
img_bw         = threshold < img_contrast;
img_bw         = create_black_holes(img_bw);





%%%================================================================

function img = create_black_holes(img)
%Eliminate all pixels located inside a circle of radius r located at the
%point center.
%This function is not being used.

m = csvread('centers_and_radii_for_elimination.csv');
center_vector = m(:,1:2);
radius_vector = m(:,3);

s = size(img);
n_rows = s(1);
n_cols = s(2);
[row_indices, col_indices] = find(img);
%Convert row and column indices to linear index
linear_indices = (col_indices-1) * n_rows + row_indices;

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











































