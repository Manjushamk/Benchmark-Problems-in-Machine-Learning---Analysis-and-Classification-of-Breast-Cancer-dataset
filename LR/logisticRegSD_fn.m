
% Logistic regression using stochastic gradient descent
% train = training data
% target = target dependent variable (only numeric !!) (i.e. 0, 1)
% learningRate = Used to limit the amount each coefficient is corrected each time it is updated.
% (same learningRate concept as in Neural nets)
% epochs = The number of times to run through the training data while updating the coefficients.
% (same epochs concept as in Neural nets)
% threshold = at what error rate we need to break the loop (lower the overfitted !)

function [ coef ] = logisticRegSD_fn (train, target, coefs, epochs,learningRate)
threshold = 0.3;
    if(isempty(coefs))
        coef = zeros(1,size(train,2)+ 1);
    else
        coef = coefs;
    end
%  for each epoch
for eachepoch = 1:1:epochs
    sumError = 0;
    for eachrow = 1:1:size(train,1)
%   calculate the probability for each row 
        yhat = predictLR_fn( train(eachrow,:),coef);
%   calculate the error
        error = target(eachrow) - yhat;
%   mean squared error
        sumError = sumError + (error^2);
%   improve the coefficient beta0 by using learningRate, error and the predicted probability
    coef(1) = coef(1) + learningRate * error * yhat * (1 - yhat);
        for i = 1:1:size(train,2)
%       improve the other coefficients (beta1, beta2, ..) by using learningRate, error, the predicted probability and the original row values
            coef(i + 1) = coef(i + 1) + learningRate * error * yhat * (1 - yhat) * train(eachrow, i);
        end
    end
%   display at each epoch !
%     fprintf('\n the current epoch is %d ',eachepoch);
%   if threshold meets break the loop;
    if(sumError < threshold)
        break
    end
end
end

        

