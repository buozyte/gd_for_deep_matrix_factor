W_hat = [-16 9 0 0; 10 5 0 0; 0 0 6 -2; 0 0 -2 4];
% W_hat = [-7 0 0 0; 0 -4 0 0; 0 0 6 0; 0 0 0 4];
% W_hat = [10 0 0 0; 0 7 0 0; 0 0 5 0; 0 0 0 -5];
[V, lambda] = eig(W_hat);

lambda = [-20 0 0 0; 0 -10 0 0; 0 0 8 0; 0 0 0 4];
% lambda = [10 0 0 0; 0 7 0 0; 0 0 5 0; 0 0 0 -5];
% lambda = [5 0 0 0; 0 3 0 0; 0 0 -1 0; 0 0 0 -5];
V = (1/sqrt(3)) * [0 1 1 1; 1 0 1 -1; 1 -1 0 1; 1 1 -1 0];
W_hat = V * lambda * V.';

eta = 10e-3;

n = 10000;
N = 3;

alpha = 0.1;
beta = alpha/2;
W_0_id = alpha * eye(4);
W_0_pert = (alpha - beta) * eye(4);

% identical init
collection = zeros(4, 4, N);
for i=1:N
    collection(:, :, i) = W_0_id;
end
collection_t = collection;

diag_elems = zeros(n+1, 4);
for step=1:n
    W = eye(4);
    for i=1:N
        W = collection(:, :, i) * W;
    end
    
    D_W = V.' * W * V;
    for i=1:4
        diag_elems(step, i) = D_W(i, i);
    end
    
    fol = eye(4);
    for j=2:N
        fol = fol * collection(:, :, j);
    end
    collection_t(:, :, 1) = collection(:, :, 1) - eta * fol.' * (W - W_hat);
    
    for i=2:(N-1)
        prev = eye(4);
        for j=1:(i-1)
            prev = prev * collection(:, :, j);
        end
        fol = eye(4);
        for j=(i+1):N
            fol = fol * collection(:, :, j);
        end
        collection_t(:, :, i) = collection(:, :, i) - eta * fol.' * (W - W_hat) * prev.';
    end
    
    prev = eye(4);
    for j=1:N-1
        prev = prev * collection(:, :, j);
    end
    collection_t(:, :, N) = collection(:, :, N) - eta * (W - W_hat) * prev.';
    
    collection = collection_t;
end
W = eye(4);
for i=1:N
    W = collection(:, :, i) * W;
end

D_W = V.' * W * V;
for i=1:4
    diag_elems(n+1, i) = D_W(i, i);
end

% -------------------------------------------------------------------------

% perturbed init
collection_pert = zeros(4, 4, N);
for i=1:N
    collection_pert(:, :, i) = W_0_id;
end
collection_pert(:, :, 1) = W_0_pert;
collection_t_pert = collection_pert;

effective_rank = zeros(n+1, 1);
approx_error = zeros(n+1, 1);

diag_elems_pert = zeros(n+1, 4);
for step=1:n
    W_pert = eye(4);
    for i=1:N
        W_pert = collection_pert(:, :, i) * W_pert;
    end
    
    D_W_pert = V.' * W_pert * V;
    for i=1:4
        diag_elems_pert(step, i) = D_W_pert(i, i);
    end
    
    fol = eye(4);
    for j=2:N
        fol = fol * collection_pert(:, :, j);
    end
    collection_t_pert(:, :, 1) = collection_pert(:, :, 1) - eta * fol.' * (W_pert - W_hat);
    
    for i=2:(N-1)
        prev = eye(4);
        for j=1:(i-1)
            prev = prev * collection_pert(:, :, j);
        end
        fol = eye(4);
        for j=(i+1):N
            fol = fol * collection_pert(:, :, j);
        end
        collection_t_pert(:, :, i) = collection_pert(:, :, i) - eta * fol.' * (W_pert - W_hat) * prev.';
    end
    
    prev = eye(4);
    for j=1:N-1
        prev = prev * collection_pert(:, :, j);
    end
    collection_t_pert(:, :, N) = collection_pert(:, :, N) - eta * (W_pert - W_hat) * prev.';
    
    collection_pert = collection_t_pert;
    
    effective_rank(step, 1) = sum(svd(W_pert)) / norm(W_pert, 2);
    approx_error(step, 1) = norm(W_hat - W_pert, 'fro');
end
W_pert = eye(4);
for i=1:N
    W_pert = collection_pert(:, :, i) * W_pert;
end

D_W_pert = V.' * W_pert * V;
for i=1:4
    diag_elems_pert(n+1, i) = D_W_pert(i, i);
end

% -------------------------------------------------------------------------

% eigenvalue recovery for different N
% tries = 6;
% n_N = 100000;
tries = 4;
n_N = 1000;
% tries = 5;
% n_N = 6000;
diag_elems_per_N = zeros(n_N+1, tries);

for N_t=1:tries
    collection_N = zeros(4, 4, N_t);
    for i=1:N_t
        collection_N(:, :, i) = W_0_id;
    end
    collection_t_N = collection_N;

    diag_elems_N = zeros(n_N+1, 4);
    for step=1:n_N
        W_N = eye(4);
        for i=1:N_t
            W_N = collection_N(:, :, i) * W_N;
        end

        D_W_N = V.' * W_N * V;
        for i=1:4
            diag_elems_N(step, i) = D_W_N(i, i);
        end

        fol = eye(4);
        for j=2:N_t
            fol = fol * collection_N(:, :, j);
        end
        collection_t_N(:, :, 1) = collection_N(:, :, 1) - eta * fol.' * (W_N - W_hat);

        for i=2:(N_t-1)
            prev = eye(4);
            for j=1:(i-1)
                prev = prev * collection_N(:, :, j);
            end
            fol = eye(4);
            for j=(i+1):N_t
                fol = fol * collection_N(:, :, j);
            end
            collection_t_N(:, :, i) = collection_N(:, :, i) - eta * fol.' * (W_N - W_hat) * prev.';
        end

        prev = eye(4);
        for j=1:N_t-1
            prev = prev * collection_N(:, :, j);
        end
        collection_t_N(:, :, N_t) = collection_N(:, :, N_t) - eta * (W_N - W_hat) * prev.';

        collection_N = collection_t_N;
    end
    W_N = eye(4);
    for i=1:N_t
        W_N = collection_N(:, :, i) * W_N;
    end

    D_W_N = V.' * W_N * V;
    for i=1:4
        diag_elems_N(n_N+1, i) = D_W_N(i, i);
    end
    
    diag_elems_per_N(:, N_t) = diag_elems_N(:, 2);
end



figure
plot(diag_elems,'LineWidth',2.0)
title('Dynamics with Identical Initialization')
xlabel('Iteration') 
ylabel("Values on the diagonal of V^{T} W^{(k)} V")
legend({'\lambda_{1} = 5', '\lambda_{2} = 3', '\lambda_{3} = -1', '\lambda_{4} = -5'})
xlim([0 n])
% ylim([-6 6])

figure
plot(diag_elems_pert,'LineWidth',2.0)
title('Dynamics with Perturbed Initialization')
xlabel('Iteration') 
ylabel("Values on the diagonal of V^{T} W^{(k)} V")
legend({'\lambda_{1} = 5', '\lambda_{2} = 3', '\lambda_{3} = -1', '\lambda_{4} = -5'})
xlim([0 n])
% ylim([-6 6])

figure
% plot(effective_rank,'LineWidth',2.0)
plot([effective_rank, diag_elems_pert],'LineWidth',2.0)
line([485 485], [-6 11], 'Color', 'black')
line([245 245], [-6 11], 'Color', 'black')
line([180 180], [-6 11], 'Color', 'black')
line([125 125], [-6 11], 'Color', 'black')
title('Effective Rank')
xlabel('Iteration') 
%ylabel("Effective rank of W^{(k)}")
legend({'Effective rank', '\lambda_{1} = 10', '\lambda_{2} = 7', '\lambda_{3} = 5', '\lambda_{4} = -5'})
xlim([0 1000])
ylim([-6 11])

figure
plot([effective_rank, approx_error],'LineWidth',2.0)
line([485 485], [0 15], 'Color', 'black')
line([245 245], [0 15], 'Color', 'black')
line([180 180], [0 15], 'Color', 'black')
line([125 125], [0 15], 'Color', 'black')
title('Approximation Error')
xlabel('Iteration') 
%ylabel("||W^{(k)} - \hat{W}||_F")
legend({'Effective rank', 'Approximation error $||W^{(k)} - \hat{W}||_F$'}, 'interpreter','latex')
xlim([0 1000])
ylim([0 15])

figure
plot(diag_elems_per_N,'LineWidth',2.0)
title('Dynamics for the Eigenvalue \lambda = 7 for N Matrices')
xlabel('Iteration') 
ylabel("Second value on the diagonal of V^{T} W^{(k)} V")
legend({'N=1', 'N=2', 'N=3', 'N=4', 'N=5'})
xlim([0 n_N])
ylim([-1 8])