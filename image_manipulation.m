function c1_tumor = image_manipulation()
close all;

c1_tumor = imread('hg_red_particles.tif');
spy(c1_tumor);
[a,b] = sort(c1_tumor(:),'descend');
% c2_tumor = imread('hg_green.tif');
% c3_tumor = imread('hg_red.tif');

fig_size   = size(c1_tumor)
n_rows     = fig_size(1);
n_cols     = fig_size(2);
if numel(fig_size) == 2
    n_channels = 1;
else
    n_channels = fig_size(3);
end

% weights = [0.299, 0.587, 0.114];
% h1 = zeros(n_rows, n_cols, 'like', c1_tumor(1,1,1));
% 
% for k = 1:n_channels
%     h1 = h1 + c1_tumor(:,:,k) * weights(k);
% end

% g1_tumor = h1;
% g2_tumor = rgb2gray(c2_tumor);
% g3_tumor = rgb2gray(c3_tumor);

h1_tumor  = histeq(c1_tumor);
% imshowpair(c1_tumor, h1_tumor);
% imhist(g1_tumor);
% pause;


% gh1_tumor = rgb2gray(h1_tumor);
% threshold = 0.5;
% bw1_tumor = im2bw(gh1_tumor, threshold);
% density   = nnz(bw1_tumor)/ numel(bw1_tumor);

figure;
f_n_rows = 2;
f_n_cols = 4;

%-------------------------------I
p = 1;
t = 'Original';
subplot(f_n_rows, f_n_cols, p);
imshow(c1_tumor);
title(t);

subplot(f_n_rows, f_n_cols, p + f_n_cols);
imhist(c1_tumor);
title(t);

%-------------------------------II
p = 2;
t = 'Contrast';
subplot(f_n_rows, f_n_cols, p);
imshow(h1_tumor);
title(t);

subplot(f_n_rows, f_n_cols, p + f_n_cols);
imhist(h1_tumor);
title(t);
return;
%-------------------------------III
p = 3;
t = 'Gray';
subplot(f_n_rows, f_n_cols, p);
imshow(gh1_tumor);
title(t);

subplot(f_n_rows, f_n_cols, p + f_n_cols);
imhist(gh1_tumor);
title(t);

%-------------------------------IV
p = 4;
t = ['BW',' L=', num2str(threshold)];
subplot(f_n_rows, f_n_cols, p);
imshow(c1_tumor);
title(t);

t = ['      \rho=', num2str(density)];
subplot(f_n_rows, f_n_cols, p + f_n_cols);
imhist(bw1_tumor);
title(t);



