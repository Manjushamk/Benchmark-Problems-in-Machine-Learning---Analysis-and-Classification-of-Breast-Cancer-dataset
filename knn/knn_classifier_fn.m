function [ classification_matrix1 ] = knn_classifier_fn( test_set,training_set,group,k,no_of_classes )


classification_matrix1=zeros(size(test_set,1),no_of_classes); %setting classification matrix to zeros with rows equal to no.of rows in test data and col = 2

for row=1:1:size(test_set,1) % 1 to number of rows in test data

clear d;
d=zeros(1,size(training_set,1)); %fills zeros in no. of columns in training set and 1 row

%euclidean Distance calculation
for j=1:1:size(training_set,1)  %size(training_set,1) is no of rows in traning set no.of rows in training set  
d(1,j)=norm(test_set(row,:)-training_set(j,:)); %norm is formula for euclidean distance
end

% % if isequal(training_set,test_set)
%     d(1,row)=9999;
% end

[d_sorted, I] = sort(d);  %d_sorted is the sorted list of d and I is the indexes
idx = I(:, 1:k); %index of the nearest neighbors

for my_idx=1:1:k
for z=1:1:no_of_classes
    if group(idx(my_idx),1)==z
        classification_matrix1(row,z)=classification_matrix1(row,z)+1;
    end
end


end

classification_matrix1(row,:)=classification_matrix1(row,:)/sum(classification_matrix1(row,:));

end
 


end

