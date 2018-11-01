%%%Source    : Houston Methodist Research Institute
%%%Location  : Houston, TX.
%%%Origin    : November 1, 2018
%%%PI        : Mauro Ferrari
%%%Supervisor: Giulia Brachi
%%%Developer : Javier Ruiz Ramirez

function process_hydrogel_image()
close all;
clc;

%Plots to be generated
plot_hydrogel = true;
plot_tumor    = true;

f_n_rows = 1;
f_n_cols = 3;

%Mapping from plot index to index in the plot
%For example, in a plot with 3 columns and 2 rows, the first three plots
%(in row 1) take the positions 1,2,3 and the subsequent 3 plots in row 2,
%take the positions 4,5,6.

map_index_to_table = @(x, level) x;
if plot_tumor + plot_hydrogel == 2
    f_n_rows = 2;
    map_index_to_table = @(x, level) f_n_cols*(level-1) + x;
end

%Load file
hydrogel_plus_tumor  = imread('hydro_gel_plus_tumor.tif');
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

contrast_interval = [0.01 0.1];

%--------------------------------------------Plot hydrogel 
if plot_hydrogel    
    [hydrogel_img, hydrogel_contrast, hydrogel_bw] = ...
        process_image_to_bw(hydrogel_img, ...
        vertical_percent, contrast_interval);
    
    tic;
    t = 'Hydro';
    subplot(f_n_rows, f_n_cols, map_index_to_table(1,1));
    % hold on;
    imshow(hydrogel_img);
    title(t);
    
    t = 'Hydro contrast';
    subplot(f_n_rows, f_n_cols, map_index_to_table(2,1));
    imshow(hydrogel_contrast)
    title(t);
    
    t = 'Hydro BW';
    subplot(f_n_rows, f_n_cols, map_index_to_table(3,1));
    imshow(hydrogel_bw)
    title(t);
    
    h = toc;
    
    disp(['Time generate plots for Hg: ', num2str(h)]);
end

%--------------------------------------------Plot tumor
if plot_tumor
    tic;
    [tumor_img, tumor_bw] = ...
        process_image_to_bw_2(tumor_img,vertical_percent);
    h = toc;
    disp(['Time to apply constraints: ', num2str(h)]);
    
    tic;
    t = 'Tumor';
    subplot(f_n_rows, f_n_cols, map_index_to_table(1,2));
    imshow(tumor_img);
    title(t);
    
    t = 'Mod. tumor BW';
    subplot(f_n_rows, f_n_cols, map_index_to_table(2,2));
    imshow(tumor_bw);
    hold on;
    spy(tumor_bw);
    xlabel('');
    ylabel('');
    axis off;
    title(t);
    
    t = 'Hg+Tumor';
    subplot(f_n_rows, f_n_cols, map_index_to_table(3,2));
    imshow(hydrogel_bw);
    hold on;
    spy(tumor_bw);
    title(t);
    xlabel('');
    
    h = toc;
    disp(['Time generate plots for Tumor: ', num2str(h)]);
    
    tic;
    nonzero_pixels_of_hydrogel = nnz(hydrogel_bw);
    nonzero_pixels_of_tumor    = nnz(tumor_bw);
    overlap                    = nnz(hydrogel_bw .* tumor_bw);
    overlap_percentage_wrt_tumor = overlap / nonzero_pixels_of_tumor * 100;
    overlap_txt = sprintf('%d', fix(overlap_percentage_wrt_tumor));
    
    fprintf('Nonzero pixels in hydrogel image: %d \n',...
        nonzero_pixels_of_hydrogel);
    fprintf('Nonzero pixels in tumor image   : %d \n',...
        nonzero_pixels_of_tumor);
    fprintf('Nonzero pixels in overlap image : %d \n',...
        overlap);
    fprintf('Overlap% with respect to tumor : %s % \n',...
        overlap_txt);
    
    t = [t, ': ', overlap_txt, '%'];
    title(t);
    
    
    h = toc;
    disp(['Time to compute overlap: ', num2str(h)]);
    
end






function [img, img_contrast, img_bw] = ...
    process_image_to_bw(img, vertical_percent, contrast_interval)
%This function modifies the hydrogel image and generates 3 images.
%The first one is the croped original image.
%The second modifies the contrast of the first one.
%The third is a black and white version of the second.

s = size(img);
width = s(2);
height= s(1);

%Crop images
starting_point = height * vertical_percent;
img            = img(starting_point:end, :);
img_contrast   = [];
img_bw         = [];

%Map intensity values (continuously) to new values in the interval [x, y]
img_contrast = imadjust(img, contrast_interval);

%Convert gray scale to black and white using Otsu's method
%Default sensitivity is set to 0.50
img_bw = imbinarize(img_contrast);




function [img, img_bw] = ...
    process_image_to_bw_2(img, vertical_percent)
% We use this function to modify the tumor image and generate two images.
%The first one is a croped version of the original.
%The second is a black and white version of the original.

s = size(img);

width = s(2);
height= s(1);

img_contrast = [];
img_bw       = [];

%Crop images
starting_point = height * vertical_percent;
img            = img(starting_point:end,:);
img_bw         = imbinarize(img);  
img_bw         = generate_and_apply_constraints(img_bw);


function img = generate_and_apply_constraints(img)
%Note that directionality is important
%When selecting the point to draw a constraint, the direction going from
%the first point to the second define a direction and a line in that
%direction.
%This function eliminates all pixels located to the left of that line,
%where left is with respect to the direction of the line.
%These lines are called constraints and one can define multiple
%constraints.
%This function has a mode called: fast_mode
%If fast_mode is active, all default constraints (located in a text file)
%are used and no graphical output is generated.
%Otherwise, the user can choose to add, or redefine the set of constraints,
%and graphical output is generated.

fast_mode = true;

if fast_mode == false
    flag = true;
    points = [];
    spy(img);
    hold on;
    set(gca, 'fontsize', 16);
    title('Current state');
    
    status = input('Use points from last run? (1=True, 0=No): ');
    
    if status == 1
        try
            points = csvread('constraint_points_for_tumor_bw_image.csv') ;
            for k = 1:2:size(points,1)
                p1 = points(k,:);
                p2 = points(k+1,:);
                
                line([p1(1), p2(1)], [p1(2), p2(2)], 'LineWidth', 2);
            end
            
            status = input('Add more constraints? (1=True, 0=No): ');
            
            if status == false
                flag = false;
            end
            
        catch
            disp('File does not exist');
            flag = true;
        end
    end
    
    
    if flag == true
        user_is_not_satisfied = true;
        
        while user_is_not_satisfied
            disp('Insert first point')
            p1 = ginput(1);
            disp(mat2str(p1))
            points = [points; p1];
            
            disp('Insert second point')
            p2 = ginput(1);
            disp(mat2str(p2))
            points = [points; p2];
            
            line([p1(1), p2(1)], [p1(2), p2(2)], 'LineWidth', 2);
            
            status = input('Add more constraints? (1=True, 0=No): ');
            
            if status == false
                user_is_not_satisfied = false;
            end
            
        end
        %Write points to file
        csvwrite('constraint_points_for_tumor_bw_image.csv', points);
        
    end
else 
    %Fast mode is active
    points = csvread('constraint_points_for_tumor_bw_image.csv') ;
end

n_points = size(points, 1);


for k = 1:2:n_points
%     disp(['Current index:', num2str(k)]);
    [row_indices, col_indices] = find(img);
    %Flip x and y coordinates to obtain true coordinates
    img_coordinates = [col_indices, row_indices].';
    n_elements      = length(row_indices);
    p1 = points(k,:);
    p2 = points(k+1,:);
    
    %     %Create line equation and plot
    %     delta = p2 - p1;
    %     line_eq = @(x) delta(2)/delta(1) * (x - p1(1)) + p1(2);
    %     tt = 1:1000:6000;
    %     xx = line_eq(tt);
    %     plot(tt,xx,'k-','linewidth',2.5);
    
    %Define line vector (constraints) (vertical)
    line_vector = (p2 - p1).';
    
    z1 = img_coordinates(1,:) - p1(1);
    z2 = img_coordinates(2,:) - p1(2);
    z  = [z1;z2];
    
    %Define a vector perpendicular to the direction of the line
    source_vector    =  z - line_vector * (line_vector.' * z);
    
    augmented_source = [source_vector; zeros(1,n_elements)];
    augmented_line   = repmat([line_vector;0], 1, n_elements);
    
    %Compute cross product between the line and the perpendicular vector.
    %We eliminate all pixels lying to the left of the line using the cross
    %product to identify the direction of the perpendicular line.
    cp = cross(augmented_line, augmented_source);
    cp = cp(3,:);
    
    mask = cp < 0;
    
    %Eliminate pixels.
    img(row_indices(mask), col_indices(mask)) = 0;

end

if fast_mode == false
    clf;
    spy(img);
end


function v = generate_block_from_xy(x,y)
%Generate a block (rectangle) of points from two points x, y
%x = [x1, x2]
%y = [y1, y2]
%This function is not being used.

x = sort(x);
y = sort(y);
x_interval = x(1):1:x(2);
y_interval = y(1):1:y(2);
[X,Y] = meshgrid(x_interval, y_interval);
v = [X(:), Y(:)];



function points = get_points_for_constraints(n_constraints)
%Get points to define a constraint.
%This function is not being used.

n_inputs = n_constraints * 2;
flag     = false;

x = input('Use points from last run? (1=True, 0=No): ');

if x == 1
    try
        points = csvread('constraint_points_for_tumor_bw_image.csv') ;
    catch
        disp('File does not exist');
        flag = true;
    end
end

if flag == true
    points = ginput(n_inputs);
    %Write points to file
    csvwrite('constraint_points_for_tumor_bw_image.csv', points);
end



function img = create_circular_profile(img, center, radius)
%Eliminate all pixels located inside a circle of radius r located at the
%point center.
%This function is not being used.

threshold =  radius;
s = size(img);
n_rows = s(1);
n_cols = s(2);
[row_indices, col_indices] = find(img);
%Convert row and column indices to linear index
linear_indices = (col_indices-1) * n_rows + row_indices;
true_linear_indices = find(img);

if ~isequal(linear_indices, true_linear_indices)
    error('Indices do not match');
end

distance = sqrt((row_indices - center(1)).^2 +...
    (col_indices - center(2)).^2);
mask = distance < threshold;
linear_indices = linear_indices(mask);
img(linear_indices) = 0;












































