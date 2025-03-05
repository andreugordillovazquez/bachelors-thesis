function addAwsData(constellationM, nodesN, signalNoiseRatio, transmissionRate, errorExponent, optimalRho)
    % Build the command string for the Python script.
    pythonScript = 'add_data.py';
    tableName = 'exponents';
    % Format simulationId and wrap in quotes for the command-line.
    simulationId = char(datetime('now', 'Format','dd-MMM-uuuu HH:mm:ss'));
    
    % Note the use of double quotes around simulationId to ensure proper handling of spaces.
    cmd = sprintf(['python3 "%s" --table %s --constellationM %s ' ...
                   '--simulationId "%s" --nodesN %s --SNR %s --transmissionRate %s ' ...
                   '--errorExponent %s --optimalRho %s'], ...
        pythonScript, tableName, num2str(constellationM), simulationId, ...
        num2str(nodesN), num2str(signalNoiseRatio), num2str(transmissionRate), ...
        num2str(errorExponent), num2str(optimalRho));
    
    % Launch the system command asynchronously on the background pool.
    f = parfeval(@system, 2, cmd);
    
    % Attach a callback that checks the command's output when done.
    % The callback function 'checkCmdResult' is defined below.
    afterEach(f, @checkCmdResult, 0);
end

function checkCmdResult(status, cmdout)
    % Callback to process the system command result.
    if status ~= 0
        % Show a warning message if the Python insertion fails.
        warning('Error inserting data. Python returned status %d:\n%s', status, cmdout);
    end
    % If successful, you may optionally log or silently ignore the output.
end