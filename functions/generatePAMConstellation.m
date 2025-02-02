function pam = generatePAMConstellation(M, d)
% GENERATE_PAM_CONSTELLATION Generates a PAM constellation.
%
%   pam_constellation = GENERATE_PAM_CONSTELLATION(M, d) 
%   generates a PAM constellation with M amplitude levels and 
%   distance 'd' between adjacent levels.
%
%   Inputs:
%       M: Constellation size (number of amplitude levels)
%       d: Distance between adjacent amplitude levels
%
%   Outputs:
%       pam_constellation: A vector containing the amplitude levels 
%                         of the PAM constellation.

    % Calculate the amplitude levels
    pam = (-((M-1)/2):((M-1)/2))*d; 

end