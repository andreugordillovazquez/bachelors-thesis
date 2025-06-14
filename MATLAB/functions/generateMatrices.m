% generateMatrices
% Computes the g_matrix and pi_matrix used for later computations in the system.
%
% Inputs:
%   X       - Vector of transmitted symbols
%   M       - Number of symbols
%   N       - Number of quadrature nodes per dimension
%   nodes   - Vector of quadrature nodes
%   weights - Vector of quadrature weights
%   SNR     - Signal-to-noise ratio
%
% Outputs:
%   g_matrix  - Matrix with computed Gaussian function values
%   pi_matrix - Matrix with corresponding weight products

function [g_matrix, pi_matrix] = generateMatrices(X, M, N, nodes, weights, SNR)
    G = @(z) exp(-abs(z).^2) / pi;    % Normalized Gaussian function
    clip = 100;                       % Clipping value to avoid NaN

    % Generate complex nodes matrix (N x N)
    z_matrix = zeros(N, N);
    for i = 1:N
        for j = 1:N
            z_matrix(i, j) = nodes(i) + 1i * nodes(j);
        end
    end

    % Initialize g_matrix (M x M*N^2)
    g_matrix = zeros(M, M * N^2);
    for j = 1:M
        d = sqrt(SNR) * (X - X(j));   % Symbol differences (Mx1)
        col_idx = (j - 1) * N^2 + (1:N^2); % Block columns for symbol j
        for i = 1:M
            g_vals = G(z_matrix + d(i));   % Gaussian values for symbol i
            g_matrix(i, col_idx) = g_vals(:).'; % Store as row vector
        end
    end
    g_matrix = max(g_matrix, exp(-clip));  % Apply clipping

    % Initialize pi_matrix (M x N^2*M)
    pi_matrix = zeros(M, N^2 * M);
    for i = 1:M
        idx = 1;
        for j = 1:N
            for k = 1:N
                pi_block = weights(j) * weights(k); % Weight product
                col = (i - 1) * N^2 + idx;         % Column index
                pi_matrix(i, col) = pi_block;       % Assign weight
                idx = idx + 1;
            end
        end
    end
end
