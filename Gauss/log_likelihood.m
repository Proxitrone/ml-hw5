function [obj] = log_likelihood(C_N, Y, N)
%LOG_LIKELIHOOD Summary of this function goes here
%   Detailed explanation goes here
    obj = -1/2 * log(det(C_N)) - 1/2 *Y'*inv(C_N)*Y - N/2*log(2*pi);
end

