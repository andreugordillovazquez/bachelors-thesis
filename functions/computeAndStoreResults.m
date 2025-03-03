% Fixed parameters for constellation and quadrature nodes
constellationM = 1024;   % Constellation size (adjust as needed)
nodesN = 20;           % Number of Gauss-Hermite nodes (adjust as needed)
d = 2;                 % PAM spacing

SNR_array = -10:20; % SNR values
transmissionRate_array = 0:0.5:10; % Transmission rates

% M-dependent data
fprintf('Performing M-dependent setup for constellationM = %d\n', constellationM);
X = generatePAMConstellation(constellationM, d);
X = X / sqrt(mean(X.^2));           % Normalize constellation energy
Q = repmat(1/constellationM, constellationM, 1);  % Uniform probability distribution

% n-dependent data
fprintf('Performing n-dependent setup for nodesN = %d\n', nodesN);
N = nodesN;
[nodes, weights] = GaussHermite_Locations_Weights(N);
z_matrix = createComplexNodesMatrix(nodes);

for i = 1:length(SNR_array)
    for j = 1:length(transmissionRate_array)
        % Simulation ID for a specific run, used as a unique ID for items with the same partition key
        simulationId = datetime('now', 'Format','yyyyMMddHHmmssSSS');

        SNR = SNR_array(i);
        transmissionRate = transmissionRate_array(j);
        fprintf('\n-----------------------------\n');
        fprintf('Processing for SNR = %.2f, Transmission Rate = %.2f\n', SNR, transmissionRate);
        
        % Define the Gaussian function for computing g_matrix
        G = @(z) exp(-abs(z).^2) / pi;
        
        % Compute matrices that depend on both M and n
        pi_matrix = createPiMatrix(constellationM, N, weights);
        g_matrix  = createGMatrix(X, z_matrix, SNR, G);
        
        % Compute derivatives at 0 and 1
        E00 = 0;
        E01 = computeEoForRhoExponential(1, Q, pi_matrix, g_matrix);
        E0P0 = computeFirstDerivativeE0(Q, pi_matrix, g_matrix, 0);
        E0P1 = computeFirstDerivativeE0(Q, pi_matrix, g_matrix, 1);
        
        % Find initial guess through Hermite Interpolation
        rho_star = demoHermiteInterpolation(E00, E0P0, E01, E0P1, transmissionRate);
        
        % Compute optimal rho and corresponding maximum E0 using Newton's method
        tol = 1e-10;  % Convergence tolerance
        if transmissionRate > E0P0
            fprintf('transmissionRate (%.6f) > derivative at 0 (%.6f): setting optimal rho = 0\n', transmissionRate, E0P0);
            optimalRho = 0;
            E0_max = E00;
        elseif transmissionRate < E0P1
            fprintf('transmissionRate (%.6f) < derivative at 1 (%.6f): setting optimal rho = 1\n', transmissionRate, E0P1);
            optimalRho = 1;
            E0_max = E01;
        else
            [optimalRho, E0_max] = optimizationNewton(Q, pi_matrix, g_matrix, transmissionRate, tol, rho_star);
        end
        
        % Error exponent
        errorExponent = E0_max - optimalRho * transmissionRate;
        fprintf('Computed optimalRho = %.10f, errorExponent = %.10f\n', optimalRho, errorExponent);
        
        % Command to run the Python script that adds data to AWS DynamoDB
        pythonScript = 'add_data.py';
        tableName = 'exponents';
        
        cmd = sprintf(['python3 "%s" --table %s ' ...
                       '--constellationM %s --simulationId "%s" --nodesN %s ' ...
                       '--SNR %s --transmissionRate %s --errorExponent %s --optimalRho %s'], ...
            pythonScript, tableName, num2str(constellationM), simulationId, ...
            num2str(nodesN), num2str(SNR), num2str(transmissionRate), ...
            num2str(errorExponent), num2str(optimalRho, '%.10f'));
        
        [status, cmdout] = system(cmd);
        if status == 0
            fprintf('Data inserted successfully for SNR = %.2f, Transmission Rate = %.2f\n', SNR, transmissionRate);
            disp(cmdout);
        else
            fprintf('Error inserting data for SNR = %.2f, Transmission Rate = %.2f. Python returned status %d\n', SNR, transmissionRate, status);
            disp(cmdout);
        end
    end
end

fprintf('\nAll computations complete.\n');