function estimatedTime = quickEstimateExecutionTime(M, N, SNR, R)

    % Load the CSV file
    T = readtable('Calculation times database - E(R).csv');
    
    % Set column names (update these if your CSV headers are different)
    progLangVar = 'ProgrammingLanguage';
    snrVar      = 'SNR';
    rateVar     = 'Rate';
    constVar    = 'Constellation_PAM_PSK_QAM_';
    M_csvVar    = 'ConstellationSize_M_';
    n_csvVar    = 'QuadratureNodesPerRealPart_n_';
    iterVar     = 'NumberOfIterations';
    timeVar     = 'TotalConsumedTime_ms_';
    
    % Filter rows: SNR equals input SNR, Rate equals R, Constellation equals 'PAM', and ProgrammingLanguage equals 'MATLAB'
    rows = (T.(snrVar) == SNR) & (T.(rateVar) == R) & ...
           strcmp(T.(constVar), 'PAM') & strcmp(T.(progLangVar), 'MATLAB');
    
    if ~any(rows)
        error('No matching data found in CSV for the given parameters.');
    end
    
    % Extract relevant columns for filtered rows
    M_csv    = T.(M_csvVar)(rows);
    n_csv    = T.(n_csvVar)(rows);
    iter_csv = T.(iterVar)(rows);
    time_csv_ms = T.(timeVar)(rows);
    
    % Compute the ns ratio for each row:
    % (TotalConsumedTime_ms_ converted to ns) divided by (M^2 * n^2 * NumberOfIterations)
    ratio = (time_csv_ms * 1e6) ./ ((M_csv.^2) .* (n_csv.^2) .* iter_csv);
    avg_ns_ratio = mean(ratio);
    avgIterations = mean(iter_csv);
    
    % Display computed averages for debugging
    fprintf('Average ns ratio: %.4f\n', avg_ns_ratio);
    fprintf('Average iterations: %.2f\n', avgIterations);
    
    % Ensure N is a column vector
    if isscalar(N)
        N = N(:);
    else
        N = N(:);
    end
    
    % Compute estimated time (in ns) for each provided N value:
    estimatedTimes_ns = avg_ns_ratio * (M^2) .* (N.^2) * avgIterations;
    totalTime_ns = sum(estimatedTimes_ns)/length(N);
    
    % Convert from ns to ms
    estimatedTime = totalTime_ns / 1e6;
    
    fprintf('Estimated total execution time: %.2f ms\n', estimatedTime);
end
