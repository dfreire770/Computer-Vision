%===================================================
% Computer Vision Programming Assignment 1 
% Prof: @Zhigang Zhu, 2003-2009  
% City College of New York

% Diego Freire, ID 0746

% The work in this assignment is my own. Any outside sources have been properly cited.

%===================================================

% ---------------- Step 1 ------------------------
% Read in an image, get information
% type help imread for more information

InputImage = 'IDPicture.bmp'; 
%OutputImage1 = 'IDPicture_bw.bmp';

C1 = imread(InputImage);
[ROWS COLS CHANNELS] = size(C1);

% ---------------- Step 2 ------------------------
% If you want to display the three separate bands
% with the color image in one window, here is 
% what you need to do
% Basically you generate three "color" images
% using the three bands respectively
% and then use [] operator to concatenate the four images
% the orignal color, R band, G band and B band

% First, generate a blank image. Using "uinit8" will 
% give you an image of 8 bits for each pixel in each channel
% Since the Matlab will generate everything as double by default
CR1 =uint8(zeros(ROWS, COLS, CHANNELS));

% Note how to put the Red band of the color image C1 into 
% each band of the three-band grayscale image CR1
for band = 1 : CHANNELS,
    CR1(:,:,band) = (C1(:,:,1));
end

% Do the same thing for G
CG1 =uint8(zeros(ROWS, COLS, CHANNELS));
for band = 1 : CHANNELS,
    CG1(:,:,band) = (C1(:,:,2));
end

% and for B
CB1 =uint8(zeros(ROWS, COLS, CHANNELS));
for band = 1 : CHANNELS,
    CB1(:,:,band) = (C1(:,:,3));
end

% Whenever you use figure, you generate a new figure window 
No1 = figure;  % Figure No. 1

%This is what I mean by concatenation
disimg = [C1, CR1;CG1, CB1]; 

% Then "image" will do the display for you!
image(disimg);

% ---------------- Step 3 ------------------------
% Now we can calculate its intensity image from 
% the color image. Don't forget to use "uint8" to 
% covert the double results to unsigned 8-bit integers

I1    = uint8(round(sum(C1,3)/3));

% You can definitely display the black-white (grayscale)
% image directly without turn it into a three-band thing,
% which is a waste of memeory space

No2 = figure;  % Figure No. 2
image(I1);

% If you just stop your program here, you will see a 
% false color image since the system need a colormap to 
% display a 8-bit image  correctly. 
% The above display uses a default color map
% which is not correct. It is beautiful, though

% ---------------- Step 4 ------------------------
% So we need to generate a color map for the grayscale
% I think Matlab should have a function to do this,
% but I am going to do it myself anyway.

% Colormap is a 256 entry table, each index has three entries 
% indicating the three color components of the index

MAP =zeros(256, 3);

% For a gray scale C[i] = (i, i, i)
% But Matlab use color value from 0 to 1 
% so I scale 0-255 into 0-1 (and note 
% that I do not use "unit8" for MAP

for i = 1 : 256,  % a comma means pause 
    for band = 1:CHANNELS,
        MAP(i,band) = (i-1)/255;
    end 
end

%call colormap to enfore the MAP
colormap(MAP);

% I forgot to mention one thing: the index of Matlab starts from
% 1 instead 0.

% Is it correct this time? Remember the color table is 
% enforced for the current one, which is  the one we 
% just displayed.

% You can test if I am right by try to display the 
% intensity image again:

No3 = figure; % Figure No. 3
image(I1);


% See???
% You can actually check the color map using 
% the edit menu of each figure window

% ---------------- Step 5 ------------------------
% Use imwrite save any image
% check out image formats supported by Matlab
% by typing "help imwrite
% imwrite(I1, OutputImage1, 'BMP');


% ---------------- Step 6 and ... ------------------------
% Students need to do the rest of the jobs from c to g.
% Write code and comments - turn it in both in hard copies and 
% soft copies (electronically)


% ---------------- Question 3 ----------------------------
% Convert the RGB image to an intensity image using the NTSC equation

I2 = 0.299 * CR1 + 0.587 * CG1 + 0.114 * CB1;

%I2 = mat2gray(I2);

% Show the image

No4 = figure;  
image(I2);


% ----------------- Question 4 --------------------
%       
% Calculate the quantization step
K = 4;
    
quantization_step = 255 / (K - 1);

% Quantize the image
img_quantized = floor(I2 / quantization_step) * quantization_step;

figure;
subplot(1, 2, 1);
imshow(I2);
title('Original Intensity Image');

subplot(1, 2, 2);
imshow(img_quantized);
title(sprintf('Quantized Intensity Image (K=%d)', K));



K = 16;
    
quantization_step = 255 / (K - 1);

% Quantize the image
img_quantized = floor(I2 / quantization_step) * quantization_step;

figure;
subplot(1, 2, 1);
imshow(I2);
title('Original Intensity Image');

subplot(1, 2, 2);
imshow(img_quantized);
title(sprintf('Quantized Intensity Image (K=%d)', K));


K = 32;
    
quantization_step = 255 / (K - 1);

% Quantize the image
img_quantized = floor(I2 / quantization_step) * quantization_step;

figure;
subplot(1, 2, 1);
imshow(I2);
title('Original Intensity Image');

subplot(1, 2, 2);
imshow(img_quantized);
title(sprintf('Quantized Intensity Image (K=%d)', K));


K = 64;
    
quantization_step = 255 / (K - 1);

% Quantize the image
img_quantized = floor(I2 / quantization_step) * quantization_step;

figure;
subplot(1, 2, 1);
imshow(I2);
title('Original Intensity Image');

subplot(1, 2, 2);
imshow(img_quantized);
title(sprintf('Quantized Intensity Image (K=%d)', K));


% ------------------------- Question 5 ------------------------

K=2;


quantization_step = 255 / (K - 1);

% Quantize the image
O_img_quantized = floor(C1 / quantization_step) * quantization_step;

figure;
subplot(1, 2, 1);
imshow(C1);
title('Original RGB Image');

subplot(1, 2, 2);
imshow(O_img_quantized);
title(sprintf('Quantized RGB Image (K=%d)', K));


K=4;

quantization_step = 255 / (K - 1);

% Quantize the image
O_img_quantized = floor(C1 / quantization_step) * quantization_step;

figure;
subplot(1, 2, 1);
imshow(C1);
title('Original RGB Image');

subplot(1, 2, 2);
imshow(O_img_quantized);
title(sprintf('Quantized RGB Image (K=%d)', K));


% ---------------------------- Question 6 ---------------------------------
% -------------- Part A

img = imread('IDPicture.bmp');

% Convert the RGB image to the range [0,1]
img = double(img) / 255;

% Extract the RGB channels
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);

% Calculate H channel
H = atan2(G - B, R - (G + B)/2) / (2*pi);
H(H < 0) = H(H < 0) + 1;

% Calculate the S channel
S = 1 - 3 * min(R, min(G, B)) ./ (R + G + B);

% Calculate the I channel
I = (R + G + B) / 3;

% Combine the H, S, and I channels into an HSI image
hsi = cat(3, H, S, I);

% Display the HSI image

imshow(hsi);
title('HSI Image');

% ----------------- Part B
% Calculate Y
Y = 0.299 * R + 0.587 * G + 0.114 * B;

% Calculate U
U = -0.14713 * R - 0.28886 * G + 0.436 * B;

% Calculate V
V = 0.615 * R - 0.51499 * G - 0.10001 * B;

YUV = cat(3,Y,U,V);

% Plot each component

figure;
subplot(2, 2, 1);
imshow(Y);
title('Y');

subplot(2, 2, 2);
imshow(U);
title('U');

subplot(2, 2, 3);
imshow(V);
title('V');

subplot(2, 2, 4);
imshow(img);
title('RGB');

% ------------------------- Question 7 -----------------------

% --------- Part A

% Read RGB image
img = imread('IDPicture.bmp');

img = im2double(img);

% Calculate the Gradient using sobel operator
sobel_filter = [-1 0 1; -2 0 2; -1 0 1];
grad_r = imfilter(img(:,:,1), sobel_filter);
grad_g = imfilter(img(:,:,2), sobel_filter);
grad_b = imfilter(img(:,:,3), sobel_filter);

% Calculate the magnitude gradient for each pixel
mag_r = sqrt(grad_r.^2 + grad_g.^2);
mag_g = sqrt(grad_g.^2 + grad_b.^2);
mag_b = sqrt(grad_b.^2 + grad_r.^2);

% find the color difference pair of adjacent pixels in the image
color_diff = sqrt(sum(diff(img, 1, 3).^2, 3));

% Combine the gradient magnitude and color difference using a weighted sum
alpha = 0.4; % adjust this parameter to control the balance between color and gradient information

edge_map = alpha * (mag_r + mag_g + mag_b) / 3 + ((1 - alpha)*color_diff) ; 
%+ (1 - alpha) * color_diff;

% Normalize
edge_map = edge_map / max(edge_map(:));

% Add a Threshold
threshold = 0.2;
edge_map_binary = edge_map > threshold;

figure;
% Show the input image and the binary edge map
imshowpair(img, edge_map_binary, 'montage');
title('Edge Map');

% Show an image overlay of the edge map over the RBG image

overlay_img = imoverlay(img, edge_map_binary, [1 0 0]);

figure;
imshow(overlay_img);
title('Sovel Image Overlay');


% --------------- Part B

% Read the input color image
img_color = imread('IDPicture.bmp');

% Convert the RGB image to grayscale
img_gray = rgb2gray(img_color);

% Calculate the edge map of the intensity image
edge_intensity = edge(img_gray, 'sobel');

% Edge maps for the R, G, and B channels
edge_red = edge(img_color(:,:,1), 'sobel');
edge_green = edge(img_color(:,:,2), 'sobel');
edge_blue = edge(img_color(:,:,3), 'sobel');

% Obtain the grayscale edge map combining the RGB edge maps
edge_gray = edge_red | edge_green | edge_blue;

% Compute the color edge map
edge_color = img_color;
edge_color(:,:,1) = edge_color(:,:,1) .* uint8(edge_red);
edge_color(:,:,2) = edge_color(:,:,2) .* uint8(edge_green);
edge_color(:,:,3) = edge_color(:,:,3) .* uint8(edge_blue);

figure;

% Display the results
subplot(2, 2, 1);
imshow(img_color);
title('Original Color Image');

subplot(2, 2, 2);
imshow(edge_intensity);
title('Intensity Edge Map');

subplot(2, 2, 3);
imshow(edge_gray);
title('Grayscale Edge Map');

subplot(2, 2, 4);
imshow(edge_color);
title('Color Edge Map');
