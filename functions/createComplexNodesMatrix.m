function z_matrix = createComplexNodesMatrix(nodes)
    N = length(nodes);
    z_matrix = zeros(N, N);
    for i = 1:N
        for j = 1:N
            z_matrix(i,j) = nodes(i) + 1i*nodes(j);
        end
    end
end