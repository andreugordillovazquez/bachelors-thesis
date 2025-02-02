function myOptimizationApp
    % Create the main floating window
    fig = uifigure('Name', 'Optimization App', 'Position', [100 100 700 500]);
    
    %% Create the Axes (for plotting) and Output Text Area
    % Define axes first so they're in scope for the callback.
    ax = uiaxes(fig, 'Position', [20 300 660 180]);
    ax.Visible = 'off';  % Hide initially
    
    % Define output text area below
    outputArea = uitextarea(fig, 'Position', [20 20 660 260]);
    
    %% Input Panel for User Parameters
    inputPanel = uipanel(fig, 'Title', 'Input Parameters', 'Position', [20 300 660 180]);
    
    % Input: N
    uilabel(inputPanel, 'Position', [20 120 100 22], 'Text', 'N:');
    editN = uieditfield(inputPanel, 'numeric', 'Position', [130 120 100 22], 'Value', 20);
    
    % Input: SNR
    uilabel(inputPanel, 'Position', [20 80 100 22], 'Text', 'SNR:');
    editSNR = uieditfield(inputPanel, 'numeric', 'Position', [130 80 100 22], 'Value', 1);
    
    % Input: M
    uilabel(inputPanel, 'Position', [20 40 100 22], 'Text', 'M:');
    editM = uieditfield(inputPanel, 'numeric', 'Position', [130 40 100 22], 'Value', 64);
    
    % Input: R
    uilabel(inputPanel, 'Position', [250 120 100 22], 'Text', 'R:');
    editR = uieditfield(inputPanel, 'numeric', 'Position', [360 120 100 22], 'Value', 0.5);
    
    % Input: tol (tolerance)
    uilabel(inputPanel, 'Position', [250 80 100 22], 'Text', 'Tolerance:');
    editTol = uieditfield(inputPanel, 'numeric', 'Position', [360 80 100 22], 'Value', 1e-7);
    
    % Option Selection via Dropdown
    uilabel(inputPanel, 'Position', [250 40 100 22], 'Text', 'Option:');
    ddOption = uidropdown(inputPanel, 'Position', [360 40 100 22], ...
        'Items', {'Optimize', 'Plot', 'Show'}, 'Value', 'Optimize');
    
    % Run Button to trigger computation
    btnRun = uibutton(inputPanel, 'push', 'Text', 'Run', ...
        'Position', [500 80 100 40], 'ButtonPushedFcn', @(btn,event) runCallback());
    
    %% Callback Function: Executes When "Run" is Clicked
    function runCallback()
        % Clear previous output and reset axes
        outputArea.Value = cell(0,1);  % Initialize as a column cell array
        cla(ax);    % Clear the axes, now accessible in this scope.
        ax.Visible = 'off';
        
        % Grab input values from the UI components
        N   = editN.Value;
        SNR = editSNR.Value;
        M   = editM.Value;
        R   = editR.Value;
        tol = editTol.Value;
        userOption = ddOption.Value;
        
        % Append a new message to the output text area
        outputArea.Value = [outputArea.Value; {sprintf('Input parameters: N=%d, SNR=%.2f, M=%d, R=%.2f, tol=%.2e', N, SNR, M, R, tol)}];
        
        % --- Prepare your computation data ---
        % (Assuming these helper functions are available in your path)
        [nodes, weights] = GaussHermite_Locations_Weights(N);
        d = 2;
        % Generate and normalize the PAM constellation
        X = generatePAMConstellation(M, d);
        X = X / sqrt(mean(X.^2));
        % Initialize probability distribution for constellation symbols
        Q = repmat(1/M, M, 1);
        % Define your Gaussian expression
        G = @(z) (1/pi) * exp(-abs(z).^2);
        % Create the required matrices for computation
        pi_matrix = createPiMatrix(M, N, weights);
        z_matrix  = createComplexNodesMatrix(nodes);
        g_matrix  = createGMatrix(X, z_matrix, SNR, G);
        
        % --- Process Based on User Option ---
        switch userOption
            case 'Optimize'
                outputArea.Value = [outputArea.Value; {'Running optimization...'}];
                
                % Initial evaluations and interpolation for an initial guess
                E00 = computeEoForRhoExponential(0, Q, pi_matrix, g_matrix);
                E01 = computeEoForRhoExponential(1, Q, pi_matrix, g_matrix);
                E0P0 = computeFirstDerivativeE0(Q, pi_matrix, g_matrix, 0);
                E0P1 = computeFirstDerivativeE0(Q, pi_matrix, g_matrix, 1);
                rho_star = demoHermiteInterpolation(E00, E0P0, E01, E0P1, R);
                
                % Newton’s Method (with boundary conditions)
                tic;
                if R > E0P0
                    rho_opt = 0;
                    E0_max = E00 - rho_opt * R;
                elseif R < E0P1
                    rho_opt = 1;
                    E0_max = E01 - rho_opt * R;
                else
                    [rho_opt, E0_max] = optimizationNewton(Q, pi_matrix, g_matrix, R, tol, rho_star);
                end
                elapsed = toc;
                
                % Display the optimization results
                outputArea.Value = [outputArea.Value; {sprintf('Optimal rho: %.6f', rho_opt)}];
                outputArea.Value = [outputArea.Value; {sprintf('Maximum E0: %.6f', E0_max)}];
                outputArea.Value = [outputArea.Value; {sprintf('Optimization Time: %.2f ms', elapsed * 1000)}];
                
            case 'Plot'
                outputArea.Value = [outputArea.Value; {'Generating plot...'}];
                % Example plot: Eo(rho)-rho*R vs. rho
                rho_vals = linspace(0, 1, 100);
                E_vals = arrayfun(@(rho) computeEoForRhoExponential(rho, Q, pi_matrix, g_matrix) - rho * R, rho_vals);
                plot(ax, rho_vals, E_vals, 'LineWidth', 2);
                xlabel(ax, 'rho');
                ylabel(ax, 'Eo(rho) - rho*R');
                title(ax, 'Optimization Plot');
                grid(ax, 'on');
                ax.Visible = 'on';
                
            case 'Show'
                outputArea.Value = [outputArea.Value; {'Showing results...'}];
                % Display some additional data (for example, the constellation and probabilities)
                outputArea.Value = [outputArea.Value; {'Constellation Points (X):'}];
                outputArea.Value = [outputArea.Value; {mat2str(X)}];
                outputArea.Value = [outputArea.Value; {'Probability Distribution (Q):'}];
                outputArea.Value = [outputArea.Value; {mat2str(Q)}];
                
            otherwise
                outputArea.Value = [outputArea.Value; {'No valid option selected.'}];
        end
    end

end
