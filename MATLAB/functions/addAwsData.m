% addAwsData
% Adds a new data entry to the AWS DynamoDB table by calling a Python script asynchronously.
%
% Inputs:
%   constellationM     - Constellation size
%   nodesN             - Number of quadrature nodes
%   signalNoiseRatio   - SNR value
%   transmissionRate   - Transmission rate
%   errorExponent      - Error exponent value
%   optimalRho         - Optimal rho value
%
% No outputs (runs asynchronously and checks result via callback)

function addAwsData(constellationM, nodesN, signalNoiseRatio, transmissionRate, errorExponent, optimalRho)
    pythonScript = 'add_data.py';                        % Python script file name
    tableName = 'exponents';                             % Table name
    simulationId = char(datetime('now', 'Format','dd-MMM-uuuu HH:mm:ss')); % Unique simulation ID

    % Build command string for Python script
    cmd = sprintf(['python3 "%s" --table %s --constellationM %s ' ...
                   '--simulationId "%s" --nodesN %s --SNR %s --transmissionRate %s ' ...
                   '--errorExponent %s --optimalRho %s'], ...
        pythonScript, tableName, num2str(constellationM), simulationId, ...
        num2str(nodesN), num2str(signalNoiseRatio), num2str(transmissionRate), ...
        num2str(errorExponent), num2str(optimalRho));

    f = parfeval(@system, 2, cmd);                       % Run command asynchronously
    afterEach(f, @checkCmdResult, 0);                    % Attach callback to check result
end

% checkCmdResult
% Callback to process the result of the system command.
function checkCmdResult(status, cmdout)
    if status ~= 0
        warning('Error inserting data. Python returned status %d:\n%s', status, cmdout);
    end
    % Optionally handle successful output here
end
