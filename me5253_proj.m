% sni_formation_sim_consistent.m
% Single-file formation sim with consistent sign convention:
% v = LM*y - P*r, controller: xdot = A_d x + B_d v, u = C_d x + D_d v, plant: ydot = -u

clear; close all; clc;

%% Parameters
M = 6;
k = 0.00001;        % start small (increase later if stable & slow)
tfinal = 150;
dt = 0.01;
tspan = 0:dt:tfinal;

%% Desired formation (hexagon radius 1) and ref
theta = (0:M-1)' * (2*pi/M);
rss_points = [cos(theta), sin(theta)];
center_ref = [1.5; 0.5];
r_ss = rss_points + center_ref';
r_x = r_ss(:,1);
r_y = r_ss(:,2);

%% Graph (ring)
A = zeros(M);
for i = 1:M
    j = mod(i,M)+1;
    A(i,j)=1; A(j,i)=1;
end
deg = sum(A,2);
L = diag(deg) - A;

% Pin only agent 1
P = diag([1 0 0 0 0 0]);
LM = L + P;

%% Controller transfer functions (scalar)
numC = [1 1 1];
denC = conv(conv([1 1],[2 1]), [1 2 5]);
numD = k * numC;
denD = conv([1 0], denC);

[A_d, B_d, C_d, D_d] = tf2ss(numD, denD);
ordD = size(A_d,1);

%% Build closed-loop A_cl & B_cl using convention:
% v = LM*y - P*r
% xdot = A_d x + B_d v
% u = C_d x + D_d v
% ydot = -u  ==> ydot = -C_d x - D_d v = -C_d x - D_d*LM*y + D_d*P*r

nCtot = M * ordD;
nY = M;
nTot = nCtot + nY;
A_cl = zeros(nTot);

% 1) controller A_d block diagonal
for i = 1:M
    idx = (i-1)*ordD + (1:ordD);
    A_cl(idx, idx) = A_d;
end

% 2) top-right: controller states receive +B_d * LM * y
TopRight = zeros(nCtot, M);
for i = 1:M
    idx = (i-1)*ordD + (1:ordD);
    TopRight(idx, :) = + B_d * LM(i,:);   % ordD x M
end
A_cl(1:nCtot, nCtot+1 : nCtot+M) = TopRight;

% 3) bottom-left: y_dot gets -C_d * x_c
for i = 1:M
    idx = (i-1)*ordD + (1:ordD);
    A_cl(nCtot + i, idx) = - C_d;   % 1 x ordD
end

% 4) bottom-right: y_dot gets -D_d * LM * y
for i = 1:M
    A_cl(nCtot + i, nCtot+1 : nCtot+M) = - D_d * LM(i,:);
end

% Build B_cl for inputs U = [r_x; r_y]
B_cl = zeros(nTot, 2*M);
for i = 1:M
    idx_c = (i-1)*ordD + (1:ordD);
    idx_y = nCtot + i;
    % controller states get -B_d * P(i,i) * r
    B_cl(idx_c, i)   = - B_d * P(i,i);    % r_x col
    B_cl(idx_c, M+i) = - B_d * P(i,i);    % r_y col
    % y_dot rows get + D_d * P(i,i) * r
    B_cl(idx_y, i)   = + D_d * P(i,i);
    B_cl(idx_y, M+i) = + D_d * P(i,i);
end

%% Initial conditions (small random around desired)
x0_x = zeros(nTot,1);
x0_y = zeros(nTot,1);
y0_x = (rand(M,1)-0.5)*0.6 + r_x;
y0_y = (rand(M,1)-0.5)*0.6 + r_y;
x0_x(nCtot + (1:M)) = y0_x;
x0_y(nCtot + (1:M)) = y0_y;

%% Stability check
eigA = eig(A_cl);
fprintf('max real(eig(A_cl)) = %.6f\n', max(real(eigA)));
% if positive -> unstable. If slightly positive, try lowering k further.

%% Simulate (use ode15s if stiff / large poles)
U_x = [r_x(:); zeros(M,1)];
odefun_x = @(t,x) A_cl * x + B_cl * U_x;
[tx, xx] = ode45(odefun_x, tspan, x0_x);

U_y = [zeros(M,1); r_y(:)];
odefun_y = @(t,x) A_cl * x + B_cl * U_y;
[ty, yy] = ode45(odefun_y, tspan, x0_y);

%% Extract outputs and compute errors
Yx = xx(:, nCtot+1 : nCtot+M);
Yy = yy(:, nCtot+1 : nCtot+M);

t = tx; dt = mean(diff(t));
vel_x = gradient(Yx, dt);
vel_y = gradient(Yy, dt);

err_x = Yx - repmat(r_x', size(Yx,1), 1);
err_y = Yy - repmat(r_y', size(Yy,1), 1);

%% Plots (trajectories, errors, rms)
figure; hold on; grid on; axis equal;
colors = lines(M);
for i=1:M
    plot(Yx(:,i), Yy(:,i),'Color',colors(i,:)); hold on;
    plot(r_ss(i,1), r_ss(i,2), 'o','MarkerFaceColor',colors(i,:));
end
title('Trajectories');

figure;
subplot(1,2,1); plot(t, err_x); title('(a) Error in x'); grid on;
subplot(1,2,2); plot(t, err_y); title('(b) Error in y'); grid on;

rms_err = sqrt(mean(err_x.^2 + err_y.^2,2));
figure; plot(t, rms_err,'k','LineWidth',1.2); grid on; title('RMS error');

%% Report final numbers
final_err = sqrt((Yx(end,:) - r_x').^2 + (Yy(end,:) - r_y').^2);
fprintf('\nFinal per-agent position errors:\n');
disp(final_err);

fprintf('\nMean steady errors (last 100 steps):\n');
mean_err_x = mean(abs(err_x(end-100:end,:)),1);
mean_err_y = mean(abs(err_y(end-100:end,:)),1);
disp([mean_err_x' mean_err_y']);
