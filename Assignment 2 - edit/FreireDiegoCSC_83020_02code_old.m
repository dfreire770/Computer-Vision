%% Author : Deepak Karuppia  10/15/01

% Modified by Diego Freire, ID 0746

% The work in this assignment is my own. Any outside sources have been properly cited.

%% Generate 3D calibration pattern: 
%% Pw holds 32 points on two surfaces (Xw = 1 and Yw = 1) of a cube 
%%Values are measured in meters.
%% There are 4x4 uniformly distributed points on each surface.

cnt = 1;

%% plane : Xw = 1

for i=0.2:0.2:0.8,
 for j=0.2:0.2:0.8,
   Pw(cnt,:) = [1 i j];
   cnt = cnt + 1;
 end
end

%% plane : Yw = 1

for i=0.2:0.2:0.8,
 for j=0.2:0.2:0.8,
   Pw(cnt,:) = [i 1 j];
   cnt = cnt + 1;
 end
end

N = cnt;

%plot3(Pw(:,1), Pw(:,2), Pw(:,3), '+');

%% Virtual camera model 

 %% Extrinsic parameters : R = RaRbRr

gamma = 40.0*pi/180.0;
Rr = [ [cos(gamma) -sin(gamma) 0];
       [sin(gamma) cos(gamma)  0];
       [  0          0         1]; ];

beta = 0.0*pi/180.0;

Rb = [ [cos(beta) 0 -sin(beta)];
       [0         1       0];
       [sin(beta) 0  cos(beta)]; ];

alpha = -120.0*pi/180.0;
Ra = [ [1      0                0];
       [0   cos(alpha)  -sin(alpha)];
       [0   sin(alpha)   cos(alpha)]; ];

R = Ra*Rb*Rr;

T = [0 0 4]';



%% Intrinsic parameters

f = 0.016;
Ox = 256;
Oy = 256;

Sx = 0.0088/512.0;
Sy = 0.0066/512.0;

Fx = f/Sx;
Fy = f/Sy;

%% asr is the aspect ratio
asr = Fx/Fy;

%% Generate Image coordinates

% surface Xw = 1
cnt = 1;
for cnt = 1:1:16,
   Pc(cnt,:) = (R*Pw(cnt,:)' + T)';
   p(cnt,:)  = [(Ox - Fx*Pc(cnt,1)/Pc(cnt,3)) (Oy - Fy*Pc(cnt,2)/Pc(cnt,3))];
end
plot(p(:,1), p(:,2), 'r+');
axis([0 512 0 512]);
grid;
hold;

% surface Yw = 1
for cnt = 17:1:32,
   Pc(cnt,:) = (R*Pw(cnt,:)' + T)';
   p(cnt,:)  = [(Ox - Fx*Pc(cnt,1)/Pc(cnt,3)) (Oy - Fy*Pc(cnt,2)/Pc(cnt,3))];
end
plot(p(17:32,1), p(17:32,2), 'g+');
%%plot3(Pc(:,1), Pc(:,2), Pc(:,3), '+');
grid;


%% Calculate Instrinsic and Extrinsic Parameters

X = Pw
x = p

% Define the number of correspondences
n = size(X, 1);

% Define the intrinsic parameter matrix

K = [Fx 0 Ox; 0 Fy Oy; 0 0 1];

% Homogenize the image projections
x_hom = [x ones(n, 1)];

% Construct the matrix A
n = size(X, 1);
x1 = x(:,1); x2 = x(:,2);
X1 = X(:,1); X2 = X(:,2); X3 = X(:,3);
O = zeros(n, 1);
I = ones(n, 1);
A = [kron([X1 O X2 O X3 O I], [1 1 1]); ...
     kron([O X1 O X2 O X3 I], [1 1 1])];

% Solve the homogeneous linear system using SVD
[~, ~, V] = svd(A);
x_est = V(:, end);
R_est = reshape(x_est(1:9), 3, 3);
T_est = x_est(10:12);

% Enforce the orthogonality constraint on R_est using SVD
[U, S, V] = svd(R_est);
R_est = U * V';

% Compute the intrinsic and extrinsic parameters
fx_est = x_est(1);
fy_est = x_est(2);
a_est = fy_est / fx_est;
ox_est = x_est(3);
oy_est = x_est(6);
alpha_est = atan2(R_est(3, 2), R_est(3, 3));
beta_est = atan2(-R_est(3, 1), sqrt(R_est(3, 2)^2 + R_est(3, 3)^2));
gamma_est = atan2(R_est(2, 1), R_est(1, 1));

% Replace the new values in the original vectors

Fx = fx_est
Fy = fy_est
asr = a_est
Ox = ox_est
Oy = oy_est

%R = R_est
%T = T_est

%alpha = alpha_est
%beta = beta_est
%gamma = gamma_est


%% Generate Image Coordinates with new Intrinsic and Extrinsic parameters

figure(2)

% surface Xw = 1
cnt = 1;
for cnt = 1:1:16,
   Pc(cnt,:) = (R_est*X(cnt,:)' + T_est)';
   p(cnt,:)  = [(ox_est - fx_est*Pc(cnt,1)/Pc(cnt,3)) (oy_est - fy_est*Pc(cnt,2)/Pc(cnt,3))];
end
plot(p(:,1), p(:,2), 'r+');
axis([0 512 0 512]);
grid;
hold;

% surface Yw = 1
for cnt = 17:1:32,
   Pc(cnt,:) = (R_est*X(cnt,:)' + T_est)';
   p(cnt,:)  = [(ox_est - fx_est*Pc(cnt,1)/Pc(cnt,3)) (oy_est - fy_est*Pc(cnt,2)/Pc(cnt,3))];
end
plot(p(17:32,1), p(17:32,2), 'g+');
%%plot3(Pc(:,1), Pc(:,2), Pc(:,3), '+');
grid;