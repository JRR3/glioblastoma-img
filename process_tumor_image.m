function process_tumor_image()
close all;

%Load file 
hydrogel_plus_tumor  = imread('hydro_gel_plus_tumor.tif');
s = size(hydrogel_plus_tumor);
disp(['Original size: ', mat2str(s)]);

%Note the inversion. The first component of the matrix represents the
%vertical directon while the second component represents the horizontal
%direction.

width = s(2);
height= s(1);

%This is the red channel and corresponds to the hydrogel
hydrogel_image = hydrogel_plus_tumor(:,:,1);

%Free memory
hydrogel_plus_tumor = [];

%This is the green channel and corresponds to the tumor
% tumor_image    = hydrogel_plus_tumor(:,:,2);

%Crop images 
starting_point = height * 0.40;
% tumor_image    = tumor_image(starting_point:end,:);
hydrogel_image = hydrogel_image(starting_point:end,:);

%Hydrogel image
hydrogel_original = hydrogel_image;

%Map intensity values (continuously) to new values in the interval [x, y]
hydrogel_contrast = imadjust(hydrogel_image,[0.01, 0.1]); 

%Convert gray scale to black and white using Otsu's method 
%Default sensitivity is set to 0.50
hydrogel_image = imbinarize(hydrogel_contrast);
%--------------------------------------------
figure;
f_n_rows = 1;
f_n_cols = 2;

p = 1;
t = 'Original';
subplot(f_n_rows, f_n_cols, p);
% hold on;
imshow(hydrogel_original);
title(t);

t = 'Modified';
subplot(f_n_rows, f_n_cols, p + 1);
imshow(hydrogel_image)
title(t);
return;

spy(c1_tumor);
[a,b] = sort(c1_tumor(:),'descend');
% c2_tumor = imread('hg_green.tif');
% c3_tumor = imread('hg_red.tif');

fig_size   = size(c1_tumor)
n_rows     = fig_size(1);
n_cols     = fig_size(2);






