% Part A

img = imread('IDPicture.bmp');

% Convert the RGB image to the range [0,1]
img = double(img) / 255;

% Extract the red, green, and blue channels
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);

% Compute the H channel
H = atan2(G - B, R - (G + B)/2) / (2*pi);
H(H < 0) = H(H < 0) + 1;

% Compute the S channel
S = 1 - 3 * min(R, min(G, B)) ./ (R + G + B);

% Compute the I channel
I = (R + G + B) / 3;

% Combine the H, S, and I channels into an HSI image
hsi = cat(3, H, S, I);

% Display the HSI image
imshow(hsi);


% Part B
Y = 0.299 * R + 0.587 * G + 0.114 * B;

U = -0.14713 * R - 0.28886 * G + 0.436 * B;

V = 0.615 * R - 0.51499 * G - 0.10001 * B;

YUV = cat(3,Y,U,V);


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



%figure(),imshow(YUV);
