% Read the input color image
img_color = imread('IDPicture.bmp');

% Convert the color image to grayscale
img_gray = rgb2gray(img_color);

% Compute the edge map of the intensity image
edge_intensity = edge(img_gray, 'sobel');

% Compute the edge maps for the R, G, and B channels
edge_red = edge(img_color(:,:,1), 'sobel');
edge_green = edge(img_color(:,:,2), 'sobel');
edge_blue = edge(img_color(:,:,3), 'sobel');

% Combine the three edge maps to obtain the grayscale edge map
edge_gray = edge_red | edge_green | edge_blue;

% Compute the color edge map
edge_color = img_color;
edge_color(:,:,1) = edge_color(:,:,1) .* uint8(edge_red);
edge_color(:,:,2) = edge_color(:,:,2) .* uint8(edge_green);
edge_color(:,:,3) = edge_color(:,:,3) .* uint8(edge_blue);

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