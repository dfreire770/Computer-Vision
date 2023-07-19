%% ===========================================================
%% CSC I6716 Computer Vision 
%% @ Zhigang Zhu, CCNY
%% Homework 4 - programming assignment: 
%% Fundamental Matrix and Feature-Based Stereo Matching
%% 
%% Name: Diego Freire
%%
%% Note: Please do not delete the commented part of the code.
%% I am going to use it to test your program
%% =============================================================

%Note: I modified part of the code adding if statements to load and save
%the points.
load_points = false;
save_points = false;

%% Read in two images 
imgl = imread('pic410.png');
imgr = imread('pic430.png');

% display image pair side by side
[ROWS COLS CHANNELS] = size(imgl);
% 
disimg = [imgl imgr];
image(disimg);

% You can change these two numbers, 
% but use the variables; I will use them
% to load my data to test your algorithms
% Total Number of control points

Nc = 12;

% Total Number of test points
Nt = 4;

% After several runs, you may want to save the point matches 
% in files (see below and then load them here, instead of 
% clicking many point matches every time you run the program

if load_points == true
    load pl.mat pl;
    load pr.mat pr;
end

%% interface for picking up both the control points and 
%% the test points

cnt = 1;
hold;

while(cnt <= Nc+Nt)

%% size of the rectangle to indicate point locations
dR = 50;
dC = 50;

%% pick up a point in the left image and display it with a rectangle....
%%% if you loaded the point matches, comment the point picking up (3 lines)%%%

if load_points == false
    disp('Select a point on the left image')
    [X, Y] = ginput(1);
    Cl = X(1); Rl = Y(1);
    pl(cnt,:) = [Cl Rl 1];
end
%% and draw it 
Cl= pl(cnt,1);  Rl=pl(cnt,2); 
rectangle('Curvature', [0 0], 'Position', [Cl Rl dC dR]);

%% and then pick up the correspondence in the right image
%%% if you loaded the point matches, comment the point picking up (three lines)%%%

if load_points == false

    disp('Select a point on the right image')
    [X, Y] = ginput(1);
    Cr = X(1); Rr = Y(1);
    pr(cnt,:) = [Cr-COLS Rr 1];

end

%% draw it
Cr=pr(cnt,1)+COLS; Rr=pr(cnt,2);
rectangle('Curvature', [0 0], 'Position', [Cr Rr dC dR]);
%plot(Cr+COLS,Rr,'r*');
drawnow;

cnt = cnt+1;
end

% This saves the the initial points

if save_points == true
    save pr.mat pr;
    save pl.mat pl;
end

%% Student work (1a) NORMALIZATION: Page 156 of the textbook and Ex 7.6
%% --------------------------------------------------------------------
%% Normalize the coordinates of the corresponding points so that
%% the entries of A are of comparable size
%% You do not need to do this, but if you cannot get correct
%% result, you may want to use this 

s=size(pl); %pr and pl have the same size

nPoints=s(1,1);

%pl_t=zeros(nPoints,3); 
%pl_t(:,3)=1;

%pr_t=zeros(nPoints,3); 
%pr_t(:,3)=1;

xmean_l = mean(pl(1,:));
ymean_l = mean(pl(2,:));

xmean_r = mean(pr(1,:));
ymean_r = mean(pr(2,:));

%xmean_l = mean([pl(1:nPoints,1)]);
%ymean_l = mean([pl(1:nPoints,2)]);

%xmean_r = mean([pr(1:nPoints,1)]);
%ymean_r = mean([pr(1:nPoints,2)]);

pl_t(:,1)=pl(:,1)-xmean_l;
pl_t(:,2)=pl(:,2)-ymean_l;

pr_t(:,1)=pr(:,1)-xmean_r;
pr_t(:,2)=pr(:,2)-ymean_r;

%
%hyp_l = sqrt((xmean_l)^2+(ymean_l)^2);
%scale_l = sqrt(2)/hyp_l;
%
%hyp_r = sqrt((xmean_r)^2+(ymean_r)^2);
%scale_r = sqrt(2)/hyp_r;
%

dist_l = sqrt(sum(pl_t.^2,1)/size(pl_t,2));
avg_dist_l = mean(dist_l);

dist_r = sqrt(sum(pr_t.^2,1)/size(pr_t,2));
avg_dist_r = mean(dist_r);

scale_l = sqrt(2) / avg_dist_l;
pl_t = scale_l * pl_t;

scale_r = sqrt(2) / avg_dist_r;
pr_t = scale_r * pr_t;

%Tl = scale_l * [1 0 -xmean_l; 0 1 -ymean_l; 0 0 1/scale_l];

%Tr = scale_r * [1 0 -xmean_r; 0 1 -ymean_r; 0 0 1/scale_r];



%% END NORMALIZATION %%

%% Student work: (1b) Implement EIGHT_POINT algorithm, page 156
%% --------------------------------------------------------------------
%% Generate the A matrix

%% Singular value decomposition of A

%% the estimate of F
% Undo the coordinate normalization if you have done normalization

A=zeros(nPoints,9);

for i=1:nPoints
  
    x1 = pl_t(i,1);
    y1 = pl_t(i,2);
    x2 = pr_t(i,1);
    y2 = pr_t(i,2);
    A(i,:) = [x1*x2 y1*x2 x2 x1*y2 y1*y2 y2 x1 y1 1];
    
end

%for i=1:nPoints
%   A(i,1)=pl_t(i,1)*pr_t(i,1);
%   A(i,2)=pl_t(i,1)*pr_t(i,2);
%   A(i,3)=pl_t(i,1);
%   A(i,4)=pl_t(i,2)*pr_t(i,1);
%   A(i,5)=pl_t(i,2)*pr_t(i,2);
%   A(i,6)=pl_t(i,2);
%   A(i,7)=pr_t(i,1);
%   A(i,8)=pr_t(i,2);
%   A(i,9)=1;
%end

% SVD of A to find F

[U,S,V]=svd(A);
f=V(:,end);
F=reshape(f,[3,3])';

% Enforce rank 2 constraint on F
[U,S,V]=svd(F);
S(3,3)=0;
FN=U*S*V';

% Denormalize

%Tl = diag([1/size(pl,2) 1/size(pl,2) 1])*[1 0 -mean(pl(1,:)); 0 1 -mean(pl(2,:)); 0 0 1];

%Tr = diag([1/size(pr,2) 1/size(pr,2) 1])*[1 0 -mean(pr(1,:)); 0 1 -mean(pr(2,:)); 0 0 1];

Tl = scale_l * [1 0 -xmean_l; 0 1 -ymean_l; 0 0 1/scale_l];

Tr = scale_r * [1 0 -xmean_r; 0 1 -ymean_r; 0 0 1/scale_r];

F = Tr'*FN*Tl;

%F_matrix=Tr'*FN*Tl;

%% END of EIGHT_POINT

%% Draw the epipolar lines for both the controls points and the test
%% points, one by one; the current one (* in left and line in right) is in
%% red and the previous ones turn into blue

%% I suppose that your Fundamental matrix is F, a 3x3 matrix

%% Student work (1c): Check the accuray of the result by 
%% measuring the distance between the estimated epipolar lines and 
%% image points not used by the matrix estimation.
%% You can insert your code in the following for loop

% Select a set of image points

points = [100 200; 150 250; 200 300; 250 350];

% Calculate the corresponding epipolar lines using the estimated fundamental matrix
lines = F * [points ones(size(points,1),1)]';

% Normalize the lines
lines = lines ./ sqrt(repmat(lines(1,:).^2 + lines(2,:).^2,3,1));

% Calculate the point-to-line distance between the image points and their corresponding epipolar lines
distances = abs(sum(lines .* [points ones(size(points,1),1)]',1));

% Calculate the mean and maximum distance
mean_distance = mean(distances);
max_distance = max(distances);

%% Plot the points and epipolar lines (the lines on the left image are slightly
% paralel to those on the right image)

cols_left = [1, 1];
cols_right = [COLS+1, 1];

for cnt=1:1:Nc+Nt,

  %Original code to plot lines (doesn't plot lines on the right side)
  %an = F*pl(cnt,:)';
  %x = 0:COLS; 
  %y = -(an(1)*x+an(3))/an(2);

  %x = x+COLS;
  %plot(pl(cnt,1),pl(cnt,2),'r*');
  %line(x,y,'Color', 'r');
  %[X, Y] = ginput(1); %% the location doesn't matter, press mouse to continue...
  %plot(pl(cnt,1),pl(cnt,2),'b*');
  %line(x,y,'Color', 'b'); 
  
  %
  an = F*pl(cnt,:)';
  
  % Plot epipolar line on left image
  x_left = [1, COLS];
  y_left = -(an(1)*x_left+an(3))/an(2);
  plot(pl(cnt,1),pl(cnt,2),'r*');
  line(x_left+cols_left(1)-1, y_left+cols_left(2)-1, 'Color', 'r');
  
  
  % Plot epipolar line on right image
  x_right = [1, COLS];
  y_right = -(an(1)*x_right+an(3))/an(2);
  plot(pr(cnt,1)+cols_right(1)-1,pr(cnt,2)+cols_right(2)-1,'b*');
  line(x_right+cols_right(1)-1, y_right+cols_right(2)-1, 'Color', 'b');
  
  [X, Y] = ginput(1); %% the location doesn't matter, press mouse to continue...
end

%% Save the corresponding points for later use... see discussions above
%save pr_t.mat pr_t;
%save pl_t.mat pl_t;

%% Save the F matrix in ascii
save F.txt F -ASCII

% Student work (1d): Find epipoles using the EPIPOLES_LOCATION algorithm page. 157
%% --------------------------------------------------------------------

[U,S,V] = svd(F);

eL = V(:,3);
eR = U(:,3);



%% save the eipoles 

%save eR.txt eRv -ASCII; 
%save eL.txt eRv -ASCII; 

% Student work (2). Feature-based stereo matching
%% --------------------------------------------------------------------
%% Try to use the epipolar geometry derived from (1) in searching  
%% correspondences along epipolar lines in Question (2). You may use 
%% a similar interface  as I did for question (1). You may use the point 
%% match searching algorithm in (1) (if you have done so), but this 
%% time you need to constrain your search windows along the epipolar lines.

% In case that the images were not loaded yet
imgl = imread('pic410.png');
imgr = imread('pic430.png');

[ROWS COLS CHANNELS] = size(imgl);

% Convert images to grayscale
grayImg1 = rgb2gray(imgl);
grayImg2 = rgb2gray(imgr);

% Extract SIFT features from left image
points1 = detectSURFFeatures(grayImg1);
[features1, validPoints1] = extractFeatures(grayImg1, points1);

% Extract SIFT features from right image
points2 = detectSURFFeatures(grayImg2);
[features2, validPoints2] = extractFeatures(grayImg2, points2);

% Find the matching points
indexPairs = matchFeatures(features1, features2, 'MaxRatio', 0.6);
matchedPoints1 = validPoints1(indexPairs(:,1));
matchedPoints2 = validPoints2(indexPairs(:,2));

% Plot the image
figure;
imshowpair(imgl, imgr, 'montage');

title('Match the Points with SIFT Features');

% Add a desired number of points
nPoints = 5;

for cnt=1:1:nPoints,

[x, y] = ginput(1); % mouse input

hold on;

plot(x,y, 'r*');

% Calculate the distances between matched points
distances = hypot(matchedPoints1.Location(:,1)-x, matchedPoints1.Location(:,2)-y);

% Find the minimal matching point distance
min_distance = min(distances);

% Use the index from the minimal distance to find the coordinates in the right Image
matchedIndex = find(distances==min(min_distance));

% Extract the location of the matched features using the index
pr=matchedPoints2(matchedIndex).Location;

% Extract coordinates

xr=pr(1);
yr=pr(2);

hold on;

%Plot Point in right image
plot(xr+COLS,yr, 'b*');

% Draw a line between selected point on the left and the matched point on
% the right image
line([x xr+COLS], [y yr], 'Color', 'g', 'LineWidth',1');

end