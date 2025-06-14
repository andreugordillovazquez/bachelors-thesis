function [maxER, optRho] = computeErrorExponent(M, N, SNR_dB, R)

    %% Convert the SNR from dB to linear
    SNR = 10^(SNR_dB/10);
    d = 2;

    %% Check if the result maxER and optRho is already present in the database
    data = getAwsData(M, N, SNR, R);

    if data.status == "success"
       disp("Data recovered from AWS");
       maxER = data.items.errorExponent;
       optRho = data.items.optimalRho;
    else
        %% Compute Gauss-Hermite Nodes and Weights
        % These are used for numerical integration over a Gaussian density.
        [nodes, weights] = GaussHermite_Locations_Weights(N);
    
        %% Generate and Normalize PAM Constellation
        % Create a PAM constellation with M symbols and spacing d.
        [X, Q] = generatePAMConstellation(M, d);
    
        %% Create Required Matrices for Computation
        [g_matrix , pi_matrix] = generateMatrices(X, M, N, nodes, weights, SNR);
    
        %% Find an Initial Guess for the Optimization
        % Compute E0 (a performance metric) and its derivative at the endpoints rho=0 and rho=1.
        E00 = 0;                                                        % E0 at rho=0
        E01 = computeEoForRhoExponential(1, Q, pi_matrix, g_matrix);    % E0 at rho=1
        E0P0 = computeFirstDerivativeE0(Q, pi_matrix, g_matrix, 0);     % Derivative at rho=0
        E0P1 = computeFirstDerivativeE0(Q, pi_matrix, g_matrix, 1);     % Derivative at rho=1
    
        %% Use Hermite interpolation to obtain a candidate starting point for optimization.
        rho_star = hermiteInterpolation(E00, E0P0, E01, E0P1, R);
    
        %% Use Newton's Method to Find the Optimal rho
        % The goal is to maximize E0(rho) - rho*R.
        tol = 1e-6;           % Convergence tolerance
        
        if R > E0P0
            % If the communication rate is higher than the derivative at 0,
            % the maximum is at rho = 0.
            optRho = 0;
            E0_max = 0;
        elseif R < E0P1
            % If the communication rate is lower than the derivative at 1,
            % the maximum is at rho = 1.
            optRho = 1;
            E0_max = computeEoForRhoExponential(optRho, Q, pi_matrix, g_matrix) - R;
        else
            % Otherwise, use Newton's method starting from rho_star to find the optimum.
            [optRho, E0_max] = optimizationNewton(Q, pi_matrix, g_matrix, R, tol, rho_star);   
        end
        
        % Compute the maximum E(R) given the optimal rho
        maxER = E0_max - optRho*R;
        
        % Add the result to AWS DynamoDB
        addAwsData(M, N, SNR, R, maxER, optRho);
    end
end