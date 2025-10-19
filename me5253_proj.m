% sni_formation_sim_fixed.m
clear; close all; clc;

%% Parameters
M = 6;                 % number of agents
k = 0.005;              % integral scaling (tunable)
tfinal = 70;          % simulation time (s)
dt = 0.01;
tspan = 0:dt:tfinal;

%% Desired formation (2D) - hexagon around origin radius 1
theta = (0:M-1)' * (2*pi/M);
rss_points = [cos(theta), sin(theta)];   % M x 2
center_ref = [1.5; 0.5];                 % reference center
r_ss = rss_points + center_ref';         % steady positions (M x 2)
r_x = r_ss(:,1);
r_y = r_ss(:,2);

%% Graph (undirected) - ring graph
A = zeros(M);
for i = 1:M
    j = mod(i,M)+1;
    A(i,j) = 1; A(j,i) = 1;
end
deg = sum(A,2);
L = diag(deg) - A;

% Pinning: pin only agent 1
P = diag([1 0 0 0 0 0]);
LM = L + P;

%% Controller transfer functions (scalar)
numC = [1 1 1];                      % s^2 + s + 1
denC = conv(conv([1 1],[2 1]), [1 2 5]); % (s+1)*(2s+1)*(s^2+2s+5)
numD = k * numC;
denD = conv([1 0], denC);  % add integrator (s term)

[A_d, B_d, C_d, D_d] = tf2ss(numD, denD);
ordD = size(A_d,1);

%% Build closed-loop system
nCtot = M*ordD;
nY = M;
nTot = nCtot + nY;
A_cl = zeros(nTot);

% Controller blocks
for i = 1:M
    idx = (i-1)*ordD + (1:ordD);
    A_cl(idx, idx) = A_d;
end

% Coupling: controller input from formation coupling LM
TopRight = zeros(nCtot, M);
for i = 1:M
    idx = (i-1)*ordD + (1:ordD);
    TopRight(idx,:) = B_d * LM(i,:);
end
A_cl(1:nCtot, nCtot+1:nCtot+M) = TopRight;

% y_dot from controller states
for i = 1:M
    idx = (i-1)*ordD + (1:ordD);
    A_cl(nCtot+i, idx) = C_d;
end

% y_dot from D_d * LM * y
for i = 1:M
    A_cl(nCtot+i, nCtot+1:nCtot+M) = D_d * LM(i,:);
end

% B_cl (reference inputs)
B_cl = zeros(nTot, 2*M);
for i = 1:M
    idx_c = (i-1)*ordD + (1:ordD);
    idx_y = nCtot + i;
    B_cl(idx_c, i)   = +B_d * P(i,i);
    B_cl(idx_c, M+i) = +B_d * P(i,i);
    B_cl(idx_y, i)   = -D_d * P(i,i);
    B_cl(idx_y, M+i) = -D_d * P(i,i);

end

%% Initial conditions
x0_x = zeros(nTot,1);
x0_y = zeros(nTot,1);
y0_x = (rand(M,1)-0.5)*0.6 + r_x;
y0_y = (rand(M,1)-0.5)*0.6 + r_y;
x0_x(nCtot + (1:M)) = y0_x;
x0_y(nCtot + (1:M)) = y0_y;

%% Simulate x-channel
U_x = [r_x(:); zeros(M,1)];
odefun_x = @(t,x) A_cl*x + B_cl*U_x;
[tx, xx] = ode45(odefun_x, tspan, x0_x);

%% Simulate y-channel
U_y = [zeros(M,1); r_y(:)];
odefun_y = @(t,x) A_cl*x + B_cl*U_y;
[ty, yy] = ode45(odefun_y, tspan, x0_y);

%% Extract outputs
Yx = xx(:, nCtot+1:nCtot+M);
Yy = yy(:, nCtot+1:nCtot+M);

%% Plot trajectories
figure('Name','Agent Trajectories');
hold on; grid on; axis equal;
colors = lines(M);
for i = 1:M
    plot(Yx(:,i), Yy(:,i), 'Color', colors(i,:), 'LineWidth', 1.5);
    plot(r_ss(i,1), r_ss(i,2), 'o', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k');
    plot(Yx(1,i), Yy(1,i), 's', 'MarkerEdgeColor', colors(i,:), 'MarkerFaceColor', 'w');
    text(r_ss(i,1)+0.03, r_ss(i,2)+0.03, sprintf('d%d', i));
end
xlabel('x (m)'); ylabel('y (m)');
title('Agent trajectories and desired steady positions');
legend(arrayfun(@(i) sprintf('Agent %d', i), 1:M, 'UniformOutput', false), 'Location', 'bestoutside');

%% Velocities and errors
t = tx; dt = mean(diff(t));
vel_x = gradient(Yx, dt);
vel_y = gradient(Yy, dt);
err_x = Yx - repmat(r_x', size(Yx,1), 1);
err_y = Yy - repmat(r_y', size(Yy,1), 1);

%% Fig.8 style: per-robot velocities
figure('Name','Per-robot velocities','Units','normalized','Position',[0.05 0.05 0.9 0.7]);
for i = 1:M
    subplot(3,2,i);
    plot(t, vel_x(:,i), 'b', 'LineWidth',1.2); hold on;
    plot(t, vel_y(:,i), 'r', 'LineWidth',1.2);
    xlabel('Time (s)'); ylabel('Velocity (m/s)');
    title(sprintf('Robot %d', i)); grid on;
    if i==1
        legend('x-dot','y-dot','Location','best');
    end
end
sgtitle('Per-robot velocities');

%% Fig.9 style: error curves
figure('Name','Formation errors','Units','normalized','Position',[0.05 0.05 0.8 0.4]);
subplot(1,2,1);
plot(t, err_x, 'LineWidth', 1.2); grid on;
xlabel('Time (s)'); ylabel('Error_x (m)'); title('(a) Error in x');
legend(arrayfun(@(i) sprintf('Robot %d', i), 1:M, 'UniformOutput', false),'Location','best');

subplot(1,2,2);
plot(t, err_y, 'LineWidth', 1.2); grid on;
xlabel('Time (s)'); ylabel('Error_y (m)'); title('(b) Error in y');
legend(arrayfun(@(i) sprintf('Robot %d', i), 1:M, 'UniformOutput', false),'Location','best');

%% RMS error convergence (clear single-curve view)
rms_err = sqrt(mean(err_x.^2 + err_y.^2, 2));
figure('Name','RMS formation error'); grid on; hold on;
plot(t, rms_err, 'k', 'LineWidth', 1.6);
xlabel('Time (s)'); ylabel('RMS position error (m)');
title('Formation error convergence');

%% Report final errors and eigenvalues
fprintf('\nSimulation finished.\n');
final_err = sqrt((Yx(end,:) - r_x').^2 + (Yy(end,:) - r_y').^2);
disp('Final per-agent error magnitudes:');
disp(final_err);

fprintf('\nMean steady errors (last 100 steps):\n');
mean_err_x = mean(abs(err_x(end-100:end,:)),1);
mean_err_y = mean(abs(err_y(end-100:end,:)),1);
disp([mean_err_x' mean_err_y']);

eigA = eig(A_cl);
fprintf('\nmax real(eig(A_cl)) = %.4f\n', max(real(eigA)));
