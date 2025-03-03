function [X, Q] = generatePAMConstellation(M, d)
% generatePAMConstellation: Generates a PAM constellation.
%
%   [X, Q] = generatePAMConstellation(M, d) generates a PAM constellation with 
%   M amplitude levels and a spacing of 'd' between adjacent levels.
%
%   Inputs:
%       M: Constellation size (number of amplitude levels)
%       d: Distance between adjacent amplitude levels
%
%   Outputs:
%       X: A normalized vector containing the amplitude levels of the PAM constellation.
%       Q: A vector representing the probability distribution of the symbols (equal probabilities).

    % Calculate the amplitude levels for a PAM constellation.
    % The levels are centered around zero, ranging from -((M-1)/2)*d to +((M-1)/2)*d.
    X = (-((M-1)/2):((M-1)/2)) * d;
    
    % Normalize the amplitude levels so that the average energy is 1.
    X = X / sqrt(mean(X.^2));
    
    % Generate the probability vector Q for the constellation.
    % Each symbol is equally likely. 
    Q = repmat(1/M, M, 1);
    % Q = Q';
end
