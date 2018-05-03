% row = each row of the testset
% coefficients = coefficients obtained after building Logistic regression model
function [ yhat ] = predictLR_fn( row, coefficients )
% formula: Yhat = beta0 + beta1 * X1 + beta2 * X2 + ..... betaN * XN
beta0 = coefficients(1);
betas = coefficients(2:size(coefficients,2));
yhat = beta0 + dot(row,betas); 
%return probability by using the below formula
% yhat = e^(beta0 + beta1 * x1 + ...) / (1 + e^(beta0 + beta1 * x1 + ..))
% can be simplified as yhat = 1/(1.0 + e^(-(beta0 + beta1 * x1 + ...)))
yhat = 1/(1 + exp(-yhat));
end