%===================================================
% Computer Vision Programming Assignment 1
% @Zhigang Zhu, 2003-2009
% City College of New York
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


% 3.
% Convert the RGB image to an intensity image using the NTSC equation
I2 = 0.299 * CR1 + 0.587 * CG1 + 0.114 * CB1;

%I2 = mat2gray(I2);

% Show the image

No4 = figure;  
image(I2);


% 4.
%       
% Calculate the quantization step
K = 4;
    
quantization_step = 255 / (K - 1);

% Quantize the image
I_quantized = floor(I2 / quantization_step) * quantization_step;

figure;
subplot(1, 2, 1);
imshow(I2);
title('Original Intensity Image');

subplot(1, 2, 2);
imshow(I_quantized);
title(sprintf('Quantized Intensity Image (K=%d)', K));



K = 16;
    
quantization_step = 255 / (K - 1);

% Quantize the image
I_quantized = floor(I2 / quantization_step) * quantization_step;

figure;
subplot(1, 2, 1);
imshow(I2);
title('Original Intensity Image');

subplot(1, 2, 2);
imshow(I_quantized);
title(sprintf('Quantized Intensity Image (K=%d)', K));



K = 32;
    
quantization_step = 255 / (K - 1);

% Quantize the image
I_quantized = floor(I2 / quantization_step) * quantization_step;

figure;
subplot(1, 2, 1);
imshow(I2);
title('Original Intensity Image');

subplot(1, 2, 2);
imshow(I_quantized);
title(sprintf('Quantized Intensity Image (K=%d)', K));


K = 64;
    
quantization_step = 255 / (K - 1);

% Quantize the image
I_quantized = floor(I2 / quantization_step) * quantization_step;

figure;
subplot(1, 2, 1);
imshow(I2);
title('Original Intensity Image');

subplot(1, 2, 2);
imshow(I_quantized);
title(sprintf('Quantized Intensity Image (K=%d)', K));


% 5.

K=2;


quantization_step = 255 / (K - 1);

% Quantize the image
O_I_quantized = floor(C1 / quantization_step) * quantization_step;

figure;
subplot(1, 2, 1);
imshow(C1);
title('Original RGB Image');

subplot(1, 2, 2);
imshow(O_I_quantized);
title(sprintf('Quantized RGB Image (K=%d)', K));


K=4;

quantization_step = 255 / (K - 1);

% Quantize the image
O_I_quantized = floor(C1 / quantization_step) * quantization_step;

figure;
subplot(1, 2, 1);
imshow(C1);
title('Original RGB Image');

subplot(1, 2, 2);
imshow(O_I_quantized);
title(sprintf('Quantized RGB Image (K=%d)', K));

%6
%6.1 RGB to HSI Image

% Convert the RGB image to the range [0,1]

R=double(CR1);
G=double(CG1);
B=double(CB1);


H = atan2(G - B, R - (G + B)/2) / (2*pi);
H(H < 0) = H(H < 0) + 1;

% Compute the S channel
S = 1 - 3 * min(R, min(G, B)) ./ (R + G + B);

% Compute the I channel
I = (R + G + B) / 3;

% Combine the H, S, and I channels into an HSI image
hsi = cat(3, H, S, I);


imshow(hsi);


%6.2 RGB to YuV Image



