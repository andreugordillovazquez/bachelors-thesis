function [g_matrix , pi_matrix] = generateMatrices(X, M, N, nodes, weights, SNR)
    % generateMatrices computes the G and π matrices used for further processing.
    % Inputs:
    %   - X: Vector of transmitted symbols.
    %   - M: Number of symbols.
    %   - N: Number of quadrature nodes per dimension.
    %   - nodes: Vector containing the quadrature nodes.
    %   - weights: Vector containing the quadrature weights.
    %   - SNR: Signal-to-noise ratio.
    % Outputs:
    %   - g_matrix: Matrix with computed Gaussian function values.
    %   - pi_matrix: Matrix with corresponding weight products.

    % Define the Gaussian function G.
    % This anonymous function computes a normalized Gaussian value for a given z.
    G = @(z) exp(-abs(z).^2) / pi;

    %% Generate the complex nodes matrix Z
    % Create an N x N matrix where each entry represents a complex node:
    % combining real parts from nodes(i) and imaginary parts from nodes(j).
    z_matrix = zeros(N, N);
    for i = 1:N
        for j = 1:N
            % Combine real and imaginary parts to form a complex number.
            z_matrix(i,j) = nodes(i) + 1i*nodes(j);
        end
    end

    %% Generate G matrix
    % Initialize g_matrix.
    % It is structured in blocks, each corresponding to a received symbol.
    % Each block has N^2 columns, covering all quadrature node combinations.
    g_matrix = zeros(M, M * N^2);  

    for j = 1:M
        % Compute the offset for the current received symbol.
        % d contains scaled differences between all symbols in X and the current symbol X(j).
        d = sqrt(SNR) * (X - X(j));  % Mx1 vector

        % Define the column indices for the current block in g_matrix.
        col_idx = (j - 1) * N^2 + (1:N^2);
        
        % Loop over source symbols to fill each row of the block.
        for i = 1:M
            % For each source symbol i, compute the Gaussian values at
            % positions given by the sum of the quadrature nodes and d(i).
            % This uses implicit expansion to add the scalar d(i) to every element of z_matrix.
            g_vals = G(z_matrix + d(i));  
            
            % Flatten the matrix into a row vector and store in the corresponding block.
            g_matrix(i, col_idx) = g_vals(:).';
        end
    end

    %% Generate π matrix
    % Initialize the π matrix.
    % It has M rows and N^2*M columns, where each block of N^2 columns corresponds
    % to the weights associated with a specific transmitted symbol.
    pi_matrix = zeros(M, N^2*M);
    
    % Process each row corresponding to a transmitted symbol.
    for i = 1:M
        % idx is used to count the current position within the N^2 block.
        idx = 1;
        % Loop over all pairs of quadrature nodes.
        for j = 1:N
            for k = 1:N
                % Compute the product of the two corresponding quadrature weights.
                pi_block = weights(j) * weights(k);
                % Determine the correct column in the π matrix for this product.
                col = (i-1) * N^2 + idx;
                % Assign the computed weight product.
                pi_matrix(i, col) = pi_block;
                % Move to the next column position in the current block.
                idx = idx + 1;
            end
        end
    end
end
