% Load the images
imgl = imread('pic410.png');
imgr = imread('pic430.png');

[ROWS COLS CHANNELS] = size(imgl);

% Convert images to grayscale
grayImg1 = rgb2gray(imgl);
grayImg2 = rgb2gray(imgr);

% Extract SIFT features from image1
points1 = detectSURFFeatures(grayImg1);
[features1, validPoints1] = extractFeatures(grayImg1, points1);

% Extract SIFT features from image2
points2 = detectSURFFeatures(grayImg2);
[features2, validPoints2] = extractFeatures(grayImg2, points2);

% Find the matching points in image2
indexPairs = matchFeatures(features1, features2, 'MaxRatio', 0.6);
matchedPoints1 = validPoints1(indexPairs(:,1));
matchedPoints2 = validPoints2(indexPairs(:,2));

% Plot the image
figure;
imshowpair(imgl, imgr, 'montage');

nPoints = 5;

for cnt=1:1:nPoints,

[x, y] = ginput(1);

hold on;

plot(x,y, 'r*');

distances = hypot(matchedPoints1.Location(:,1)-x, matchedPoints1.Location(:,2)-y);

min_distance = min(distances);

matchedIndex = find(distances==min(min_distance));

pr=matchedPoints2(matchedIndex).Location;

xr=pr(1);
yr=pr(2);

hold on;

plot(xr+COLS,yr, 'b*');

%showMatchedFeatures(grayImg1, grayImg2, matchedPoints1, matchedPoints2, 'montage');
%title('Matched Points');

line([x xr+COLS], [y yr], 'Color', 'y');

end