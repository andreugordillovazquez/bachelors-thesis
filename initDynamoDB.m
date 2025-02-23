%% initDynamoDB.m
% This script initializes the MATLAB interface to Amazon DynamoDB.
% It adds the necessary paths, creates and initializes the DynamoDB client,
% and verifies connectivity by listing the available tables.

% https://eu-north-1.console.aws.amazon.com/dynamodbv2/home?region=eu-north-1#tables

% Add the AWS DynamoDB MATLAB interface to the path
% (Change the folder path below to match your installation location.)
addpath(genpath('/Users/andreugordillovazquez/Desktop/UPF/TFG/mathworks-aws-support/matlab-aws-dynamodb'));

% (Optional) Refresh MATLAB's cache to recognize new files
rehash toolboxcache;

% Create the DynamoDB client object
fprintf('Creating DynamoDB client...\n');
ddb = aws.dynamodbv2.AmazonDynamoDBClient;

% Initialize the client (this will use your default credentials and region)
fprintf('Initializing DynamoDB client...\n');
ddb.initialize();

% Verify connectivity by listing tables.
% Create a ListTablesRequest object
req = aws.dynamodbv2.model.ListTablesRequest();
try
    listResult = ddb.listTables(req);
    tableNames = listResult.getTableNames();
    fprintf('DynamoDB Tables:\n');
    disp(tableNames);
catch ME
    fprintf('Error listing tables: %s\n', ME.message);
end

% (Optional) Make the client available in the base workspace
assignin('base', 'ddb', ddb);

fprintf('DynamoDB client initialized successfully.\n');