function [ prob ] = my_svm_test_fn( model, test_input, no_of_labels )

label_vector1=ones(size(test_input,1),1);
%# Get probability estimates of test instances using each model
prob                    = zeros(size(test_input,1),no_of_labels);
for k=1:no_of_labels
    [~,~,p]                 = svmpredict(ones(length(label_vector1),1), test_input, model{k}, '-b 1');
    prob(:,k)               = p(:,model{k}.Label==1);    % Probability of class==k
end

end

