function pi_matrix = createPiMatrix(M, N, weights)
    % Initialize matrix with zeros
    pi_matrix = zeros(M, N^2*M);
    
    % For each row
    for i = 1:M
        % For all combinations of weights
        idx = 1;
        for j = 1:N
            for k = 1:N
                % Calculate weight product
                pi_block = weights(j)*weights(k);
                % Calculate column position
                col = (i-1)*N^2 + idx;
                % Assign the weight product
                pi_matrix(i, col) = pi_block;
                idx = idx + 1;
            end
        end
    end
end