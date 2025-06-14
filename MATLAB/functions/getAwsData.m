% getAwsData
% Calls a Python script to retrieve data from an AWS table for given parameters.
%
% Inputs:
%   M  - Constellation size
%   N  - Number of quadrature nodes
%   SNR - Signal-to-noise ratio
%   R  - Transmission rate
%
% Output:
%   data - Structure with the retrieved data or error information

function data = getAwsData(M, N, SNR, R)
    pythonScript = 'get_data.py';    % Python script file name
    tableName = 'exponents';         % Table name for the script

    % Build command string to call the Python script with parameters
    cmd = sprintf('python3 "%s" --table %s --constellationM %f --nodesN %f --SNR %f --transmissionRate %f', ...
        pythonScript, tableName, M, N, SNR, R);

    [status, cmdout] = system(cmd);  % Execute the command

    if status == 0
        try
            data = jsondecode(cmdout);   % Decode JSON output
        catch ME
            data = struct('status', 'error', 'message', ['JSON decoding failed: ' ME.message]);
        end
    else
        data = struct('status', 'error', 'message', sprintf('Python script call failed with status %d', status));
    end
end
