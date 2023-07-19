% Read the input RGB image
rgb_img = imread('IDPicture.bmp');

% Convert the image to double precision for processing
rgb_img = im2double(rgb_img);

% Compute the gradient magnitude for each color channel using the Sobel operator
sobel_filter = [-1 0 1; -2 0 2; -1 0 1];
grad_r = imfilter(rgb_img(:,:,1), sobel_filter);
grad_g = imfilter(rgb_img(:,:,2), sobel_filter);
grad_b = imfilter(rgb_img(:,:,3), sobel_filter);

% Compute the magnitude of the gradient at each pixel in each color channel
mag_r = sqrt(grad_r.^2 + grad_g.^2);
mag_g = sqrt(grad_g.^2 + grad_b.^2);
mag_b = sqrt(grad_b.^2 + grad_r.^2);

%figure;
%imshow(mag_g);

% Compute the color difference between each pair of adjacent pixels in the image
color_diff = sqrt(sum(diff(rgb_img, 1, 3).^2, 3));

% Combine the gradient magnitude and color difference using a weighted sum
alpha = 0.4; % adjust this parameter to control the balance between color and gradient information

edge_map = alpha * (mag_r + mag_g + mag_b) / 3 + ((1 - alpha)*color_diff) ; 
%+ (1 - alpha) * color_diff;

% Normalize the resulting image to the range [0, 1]
edge_map = edge_map / max(edge_map(:));

% Threshold the normalized image to obtain the final binary edge map
threshold = 0.2;
edge_map_binary = edge_map > threshold;

%figure(2)
% Display the input image and the binary edge map side by side
imshowpair(rgb_img, edge_map_binary, 'montage');


overlay_img = imoverlay(rgb_img, edge_map_binary, [1 0 0]);
figure;
imshow(overlay_img);