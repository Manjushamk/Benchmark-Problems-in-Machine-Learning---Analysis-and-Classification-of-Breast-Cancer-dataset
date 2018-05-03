clear
clc
accuracies_for_various_runs=0;
%loading of dataset
data=load('breastcancer.txt');
%defining the patterns
class_label=10;
no_of_folds=10;
% k value for the K nearest neighbor inthis case k is 3
k=3;
%no_of_classes 
no_of_classes=length(unique(data(:,class_label)));
[size_x size_y]=size(data); 
count_in_each_fold=ceil(size_x/no_of_folds);

%fold is for classifying the test and training data
for i=1:1:no_of_folds
    if i==no_of_folds
    fold(i).indices=(i-1)*count_in_each_fold+1:size_x;   
    else
    fold(i).indices=(i-1)*count_in_each_fold+1:i*count_in_each_fold;
    end
end
for i=1:1:no_of_folds
        length1=0;
        for j=1:1:no_of_folds
            if j==i
                fold(i).testing_indices=fold(j).indices;            
            else
                length2=length(fold(j).indices);    
                fold(i).training_indices(length1+1:length1+length2)=fold(j).indices;
                length1=length1+length2;
            end
        end

end


for round=1:3
% each fold for the
for my_fold=1:1:no_of_folds
training_data=data(fold(my_fold).training_indices,:);
test_data=data(fold(my_fold).testing_indices,:);
class_vector=training_data(:,class_label);
class_vector1=test_data(:,class_label);
        target_training_class_matrix=zeros(size(training_data,1),no_of_classes);
        for i=1:1:size(training_data,1)
        target_training_class_matrix(i,class_vector(i,1))=1;
        end
        
        
        
test_class_confidence_matrix= knn_classifier_fn(test_data(:,1:size(test_data,2)-1),training_data(:,1:size(training_data,2)-1),training_data(:,class_label),k,no_of_classes);

                clear class_vector1_obtained
                for row=1:1:size(test_data,1)
                class_vector1_obtained(row,1)=find(test_class_confidence_matrix(row,:)==max(test_class_confidence_matrix(row,:)),1);
                end
                correct=0;
                chk=[class_vector1 class_vector1_obtained];
                for i=1:1:size(chk,1)
                    if chk(i,1)==chk(i,2)
                        correct=correct+1;
                    end
                end
                accuracy(my_fold)=correct*100/size(chk,1);      
                
end
accu(round)=sum(accuracy)/10;
end
Accuracy_of_classification =sum(accu)/3;
fprintf('\n Accuracy of test data(breast Cancer) is %f \n',Accuracy_of_classification);

