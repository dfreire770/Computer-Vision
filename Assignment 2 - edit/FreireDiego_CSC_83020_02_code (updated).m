%% Author : Deepak Karuppia  10/15/01

% Modified by Diego Freire

% The work in this assignment is my own. Any outside sources have been properly cited.

%% Generate 3D calibration pattern: 
%% Pw holds 32 points on two surfaces (Xw = 1 and Yw = 1) of a cube 
%% Values are measured in meters.
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

gamma = 40*pi/180.0;
Rr = [ [cos(gamma) -sin(gamma) 0];
       [sin(gamma) cos(gamma)  0];
       [  0          0         1]; ];

beta = 0*pi/180.0;

Rb = [ [cos(beta) 0 -sin(beta)];
       [0         1       0];
       [sin(beta) 0  cos(beta)]; ];

alpha = -120*pi/180.0;
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

% asr is the aspect ratio
asr = Fx/Fy;

%% Generate Image coordinates

%% surface Xw = 1
cnt = 1;
for cnt = 1:1:16
   Pc(cnt,:) = (R*Pw(cnt,:)' + T)';
   p(cnt,:)  = [(Ox - Fx*Pc(cnt,1)/Pc(cnt,3)) (Oy - Fy*Pc(cnt,2)/Pc(cnt,3))];
end
plot(p(:,1), p(:,2), 'r+');
axis([0 512 0 512]);
hold;

% surface Yw = 1
for cnt = 17:1:32,
   Pc(cnt,:) = (R*Pw(cnt,:)' + T)';
   p(cnt,:)  = [(Ox - Fx*Pc(cnt,1)/Pc(cnt,3)) (Oy - Fy*Pc(cnt,2)/Pc(cnt,3))];
end
plot(p(17:32,1), p(17:32,2), 'g+');
grid;

%% Camera Calibration

X = Pw(:,1);
Y = Pw(:,2);
Z = Pw(:,3);

u = p(:, 1) - Ox;
v = p(:, 2) - Oy;

% Build matrix A
A = [u .* Pw(:, 1), u .* Pw(:, 2), u .* Pw(:, 3), u, -v .* Pw(:, 1), -v .* Pw(:, 2), -v .* Pw(:, 3), -v];

% Get solution v
[~, ~, V] = svd(A);
%V = V.';
v = V(:,end);

% Determine scale
scale = norm(v(1:3));

% Determine aspect ratio
v_sqr = v.^2;
asr = sqrt(v_sqr(5) + v_sqr(6) + v_sqr(7)) / norm(v(1:3));

% Determine 1st two rows of R

R = zeros(3,3);
R(1,1:3) = v(5:7)/(scale*asr);
R(2,1:3) = v(1:3)/scale;
R(3,1:3) = cross(R(1,:).',R(2,:).');

% Determine 1st two components of T

T = zeros(3,1);
T(1) = v(8)/(scale*asr);
T(2) = v(4)/scale;

% Determine sign s

Xc = R(1,1)*Pw(1,1) + R(1,2)*Pw(1,2) + R(1,3)*Pw(1,3) + T(1);
xc = p(1,1);

s = sign(Xc*xc);

% Update sign using s

R(1,1:3) = s*R(1,1:3);
R(2,1:3) = s*R(2,1:3);

T(1) = s*T(1);
T(2) = s*T(2);

% Compute 3rd row of R

R3_t = cross(R(2,:), R(1,:));
R(3,:) = R3_t;


% Enforce orthogonality on R

[U,D,V2] = svd(R);
R = U*V2.';

% Solve for Tz and fx and fy

A2 = [u, X*R(1,1) + Y*R(1,2) + Z*R(1,3) + T(1)];
B2 = -u.*(X*R(3,1) + Y*R(3,2) + Z*R(3,3));

sol = A2 \ B2;

T(3) = sol(1);
fx = sol(2);
inv_asr = 1/asr;
fy = fx * inv_asr;

ox = 256;
oy = 256;

% Find Projective Matrix

intrinsic_matrix = [-fx 0 ox; 0 -fy oy; 0 0 1];
extrinsic_matrix = [R(1,1) R(1,2) R(1,3) T(1); R(2,1) R(2,2) R(2,3) T(2); R(3,1) R(3,2) R(3,3) T(3)];
M = intrinsic_matrix*extrinsic_matrix;

% Reproject
world_points = [X Y Z];
homogeneous_points = [world_points ones(size(world_points,1),1)];

reprojected_points = M * homogeneous_points.';
normalized_points = zeros(2, size(reprojected_points,2));
normalized_points(1,:) = reprojected_points(1,:) ./ reprojected_points(3,:);
normalized_points(2,:) = reprojected_points(2,:) ./ reprojected_points(3,:);
normalized_points = normalized_points.';

% Plot results
figure;
plot(normalized_points(:,1), normalized_points(:,2), 'b*');
grid;
axis([0 512 0 512]);

%% Noisy image

% Add noise randomly

Pw = Pw + 0.0001*randn(32,3);
p_n= p + 0.5*randn(32,2);   

X = Pw(:,1);
Y = Pw(:,2);
Z = Pw(:,3);

u = p_n(:, 1)  - Ox;
v = p_n(:, 2) - Oy;

% Build matrix A
A = [u .* Pw(:, 1), u .* Pw(:, 2), u .* Pw(:, 3), u, -v .* Pw(:, 1), -v .* Pw(:, 2), -v .* Pw(:, 3), -v];

% Get solution v
[~, ~, V] = svd(A);
%V = V.';
v = V(:,end);

% Determine scale
scale = norm(v(1:3));

% Determine aspect ratio
v_sqr = v.^2;
asr = sqrt(v_sqr(5) + v_sqr(6) + v_sqr(7)) / norm(v(1:3));

% Determine 1st two rows of R

R = zeros(3,3);
R(1,1:3) = v(5:7)/(scale*asr);
R(2,1:3) = v(1:3)/scale;
R(3,1:3) = cross(R(1,:).',R(2,:).');

% Determine 1st two components of T

T = zeros(3,1);
T(1) = v(8)/(scale*asr);
T(2) = v(4)/scale;

% Determine sign s

Xc = R(1,1)*Pw(1,1) + R(1,2)*Pw(1,2) + R(1,3)*Pw(1,3) + T(1);
xc = p(1,1);

s = sign(Xc*xc);

% Recalculate using sign s

R(1,1:3) = s*R(1,1:3);
R(2,1:3) = s*R(2,1:3);

T(1) = s*T(1);
T(2) = s*T(2);

% Compute 3rd row of R

R3_t = cross(R(2,:), R(1,:));
R(3,:) = R3_t;


% Enforce orthogonality constraint on R

[U,D,V2] = svd(R);
R = U*V2.';

% Solve for Tz and fx and fy

A2 = [u, X*R(1,1) + Y*R(1,2) + Z*R(1,3) + T(1)];
B2 = -u.*(X*R(3,1) + Y*R(3,2) + Z*R(3,3));

sol = A2 \ B2;

T(3) = sol(1);
fx = sol(2);
inv_asr = 1/asr;
fy = fx * inv_asr;

ox = 256;
oy = 256;

% Find Projective Matrix

intrinsic_matrix = [-fx 0 ox; 0 -fy oy; 0 0 1];
extrinsic_matrix = [R(1,1) R(1,2) R(1,3) T(1); R(2,1) R(2,2) R(2,3) T(2); R(3,1) R(3,2) R(3,3) T(3)];
M = intrinsic_matrix*extrinsic_matrix;

% Reproject
world_points = [X Y Z];
homogeneous_points = [world_points ones(size(world_points,1),1)];

reprojected_points = M * homogeneous_points.';
normalized_points = zeros(2, size(reprojected_points,2));
normalized_points(1,:) = reprojected_points(1,:) ./ reprojected_points(3,:);
normalized_points(2,:) = reprojected_points(2,:) ./ reprojected_points(3,:);
normalized_points = normalized_points.';

% Plot results
figure;
plot(normalized_points(:,1), normalized_points(:,2), 'b*');
grid;
axis([0 512 0 512]);
