function [grad_i] = log_likelihood_grad_i(C_N, C_N_grad_i, Y)
%LOG_LIKELIHOOD_GRAD_I Summary of this function goes here
%   Detailed explanation goes here
    
    grad_i = -1/2*trace(inv(C_N)*C_N_grad_i) + 1/2* Y'*inv(C_N)*C_N_grad_i*inv(C_N)*Y;
end

