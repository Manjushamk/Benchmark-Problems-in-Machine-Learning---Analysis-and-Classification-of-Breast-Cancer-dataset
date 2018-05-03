function [ model, prob ] = my_svm_train_fn( train_input, label_vector, no_of_labels)


model                   = cell(no_of_labels,1);
for k=1:no_of_labels
    model{k}                = svmtrain(double(label_vector==k), train_input, '-c 1 -g 0.2 -b 1');
end

prob                    = zeros(size(train_input,1),no_of_labels);
for k=1:no_of_labels
    [~,~,p1]                 = svmpredict(double(label_vector==k), train_input, model{k}, '-b 1');
    prob(:,k)               = p1(:,model{k}.Label==1);    % Probability of class==k
end


end

