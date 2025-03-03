function g_matrix = createGMatrix(X, z_matrix, SNR, G)
    M = length(X);              % Number of source symbols
    N = size(z_matrix, 1);      % Assuming z_matrix is N-by-N
    g_matrix = zeros(M, M * N^2);  % Initialize matrix: one block per received symbol j

    % SNR = 10^(SNR/10);

    % Loop over received symbols (j) and source symbols (i)
    disp(["SNR: ", SNR]);
    for j = 1:M
        % Precompute the term for the given j: sqrt(SNR)*(X - X(j))
        d = sqrt(SNR) * (X - X(j));  % This is an Mx1 vector; each entry corresponds to a different i
        
        % Define the column indices for the current block corresponding to j
        col_idx = (j - 1) * N^2 + (1:N^2);
        
        % Loop over source symbols (i)
        for i = 1:M
            % For each (i, j), compute the G function over all quadrature nodes.
            % Here, z_matrix is N-by-N and d(i) is a scalar. With implicit expansion,
            % we add d(i) to every element of z_matrix.
            g_vals = G(z_matrix + d(i));  
            
            % Vectorize the result and store it in the appropriate block of g_matrix.
            g_matrix(i, col_idx) = g_vals(:).';
        end
    end
end
