function userInput = getUserInputUI()
    % getUserInputUI creates a single-window UI to collect input parameters.
    % It returns a structure 'userInput' with fields:
    %   N   - Number of quadrature nodes
    %   SNR - Signal-to-Noise Ratio
    %   M   - Constellation Size
    %   R   - Transmission Rate
    %   tol - Tolerance

    % Initialize the output structure.
    userInput = struct('N', [], 'SNR', [], 'M', [], 'R', [], 'tol', []);

    % Create a modal UI figure
    fig = uifigure('Name', 'Input Parameters', ...
                   'Position', [500 500 400 300], 'Resize', 'off');

    % --- Input for the number of quadrature nodes ---
    uilabel(fig, 'Text', 'Number of quadrature nodes:', ...
            'Position', [20 250 180 22]);
    editNodes = uieditfield(fig, 'numeric', 'Position', [210 250 150 22], 'Value', 20);

    % --- Input for the SNR ---
    uilabel(fig, 'Text', 'Signal-to-Noise Ratio (SNR):', ...
            'Position', [20 210 180 22]);
    editSNR = uieditfield(fig, 'numeric', 'Position', [210 210 150 22], 'Value', 1);

    % --- Input for the transmission rate ---
    uilabel(fig, 'Text', 'Transmission Rate:', ...
            'Position', [20 170 180 22]);
    editRate = uieditfield(fig, 'numeric', 'Position', [210 170 150 22], 'Value', 0.5);

    % --- Dropdown for the constellation size ---
    uilabel(fig, 'Text', 'Constellation Size:', ...
            'Position', [20 130 180 22]);
    % Options as strings (they will be converted to numeric later)
    constellationOptions = {'2','4','8','16','32','64','128','256','512','1024'};
    popupConst = uidropdown(fig, 'Position', [210 130 150 22], ...
                            'Items', constellationOptions, 'Value', '64');

    % --- Dropdown for the tolerance ---
    uilabel(fig, 'Text', 'Tolerance:', ...
            'Position', [20 90 180 22]);
    toleranceOptions = {'1e-4','1e-7','1e-10'};
    popupTol = uidropdown(fig, 'Position', [210 90 150 22], ...
                          'Items', toleranceOptions, 'Value', '1e-4');

    % --- Submit button ---
    uibutton(fig, 'push', 'Text', 'Confirm', ...
        'Position', [150 30 100 30], ...
        'ButtonPushedFcn', @(btn,event) submitCallback());

    % Block execution until the user clicks Submit
    uiwait(fig);

    % --- Callback function for the Submit button ---
    function submitCallback()
        % Read values from the UI components and store in the output structure.
        userInput.N   = editNodes.Value;                   % Quadrature nodes
        userInput.SNR = editSNR.Value;                     % SNR
        userInput.R   = editRate.Value;                    % Transmission rate
        userInput.M   = str2double(popupConst.Value);      % Constellation size
        userInput.tol = str2double(popupTol.Value);        % Tolerance

        % Resume execution and close the UI figure.
        uiresume(fig);
        delete(fig);
    end
end