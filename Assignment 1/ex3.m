% Load an RGB image into a MATLAB matrix
rgb_image = imread('IDPicture.bmp');

[ROWS, COLS, CHANNELS] = size(rgb_image);

% Extract the red, green, and blue channels of the image
%red_channel = rgb_image(:, :, 1);
%green_channel = rgb_image(:, :, 2);
%blue_channel = rgb_image(:, :, 3);

red_channel =uint8(zeros(ROWS, COLS, CHANNELS));

for band = 1 : CHANNELS,
    red_channel(:,:,band) = (rgb_image(:,:,1));
end

green_channel =uint8(zeros(ROWS, COLS, CHANNELS));
for band = 1 : CHANNELS,
    green_channel(:,:,band) = (rgb_image(:,:,2));
end

blue_channel =uint8(zeros(ROWS, COLS, CHANNELS));
for band = 1 : CHANNELS,
    blue_channel(:,:,band) = (rgb_image(:,:,3));
end


% Apply the NTSC equation to convert the RGB image to an intensity image
intensity_image = 0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel;

% Scale the intensity image so that its minimum value is 0 and its maximum value is 255
%intensity_image = intensity_image - min(intensity_image(:));
%intensity_image = 255 * intensity_image / max(intensity_image(:));

% Convert the intensity image to 8-bit unsigned integer format
intensity_image = uint8(intensity_image);

% Save the intensity image to a file
%imwrite(intensity_image, 'intensity_image.png');
No4 = figure; % Figure No. 4
image(intensity_image);
