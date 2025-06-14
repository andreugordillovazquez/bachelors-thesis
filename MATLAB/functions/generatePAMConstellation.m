% generatePAMConstellation
% Generates a normalized PAM constellation and its probability distribution.
%
% Inputs:
%   M - Constellation size (number of amplitude levels)
%   d - Distance between adjacent amplitude levels
%
% Outputs:
%   X - Normalized vector of amplitude levels (1×M)
%   Q - Probability distribution vector (M×1), equal probabilities

function [X, Q] = generatePAMConstellation(M, d)
    X = (-((M-1)/2):((M-1)/2)) * d;      % Amplitude levels
    X = X / sqrt(mean(X.^2));            % Normalize to unit average energy
    Q = repmat(1/M, M, 1);               % Equal probability for each symbol
end
