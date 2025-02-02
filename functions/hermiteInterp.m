function [p, dp] = hermiteInterp(A, B, C, D)
% hermiteInterp  Constructs a cubic Hermite interpolant P(rho) on [0, 1]
%   matching:
%       P(0)   = A
%       P'(0)  = B
%       P(1)   = C
%       P'(1)  = D
%
% INPUTS:
%   A = E0(0)
%   B = E0'(0)
%   C = E0(1)
%   D = E0'(1)
%
% OUTPUTS:
%   p(rho)   - function handle for P(rho)
%   dp(rho)  - function handle for P'(rho)

    % Coefficients for the cubic P(rho) = a + b*rho + c*rho^2 + d*rho^3
    a = A;
    b = B;
    
    % Solve for c, d from the endpoint/derivative constraints
    % Using the standard Hermite interpolation formula for 2 points:
    %   c = 3(C - A) - 2B - D
    %   d = -2(C - A) + B + D
    
    c = 3*(C - A) - 2*B - D;
    d = -2*(C - A) + B + D;

    % Create function handles for the polynomial and its derivative
    p  = @(rho) a + b*rho + c*rho.^2 + d*rho.^3;
    dp = @(rho) b + 2*c*rho + 3*d*rho.^2;
end
