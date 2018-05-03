clearvars -except accuracies
ordered_data=load('breastcancer.txt');
[size_x size_y]=size(ordered_data);
accuracy = 0;
class_label= 10;
epochs = 500;
rounds = 5;
no_of_classes=length(unique(ordered_data(:,class_label)));
learningRate=0.7;%input('enter learning rate of neural network classifeir.. \n 0.7 is preferable\n');
%randomize the rows of the dataset
for r=1:1:rounds
tp = 0;
tn = 0;
fp = 0;
fn = 0;
data=ordered_data(randperm(size(ordered_data,1)),:);

for j=1:1:size_y
min_data(j)=min(data(:,j));
max_data(j)=max(data(:,j));
end;

%normalisation
for i=1:1:size_x
for j=1:1:size_y-1
    data(i,j)=(data(i,j)-min_data(j))/(max_data(j)-min_data(j));
end;
end;


train = data(1:floor(0.7*size_x),:);
trainTarget = train(:,size_y);
train = train(:,1:size_y-1);
test = data(floor(0.7*size_x)+1:size(data,1),:);
testTarget = test(:, size_y);
test = test(:, 1:size_y-1);

for i=1:1:length(trainTarget)
	if trainTarget(i) ~= 1
		trainTarget(i) = 0;
	end
end

for i=1:1:length(testTarget)
	if testTarget(i) ~= 1
		testTarget(i) = 0;
	end
end

coefs = zeros(1,size(train,2)+ 1);
coefs = logisticRegSD_fn(train, trainTarget, coefs , epochs , learningRate);

probs = [];
for i=1:1:length(testTarget)
	probs = [probs, predictLR_fn(test(i, :), coefs)];
end

for i=1:1:length(probs)
	if probs(i) >= 0.5
		probs(i) = 1;
	else
		probs(i) = 0;
	end
end

%for confusion matrix
for i=1:1:length(probs)
	if probs(i) == 0 && testTarget(i) == 0
		tn = tn + 1;
	elseif probs(i) == 1 && testTarget(i) == 1
		tp = tp + 1;
	elseif probs(i) == 1 && testTarget(i) == 0
		fp = fp + 1;
	else
		fn = fn + 1;
	end
end
fprintf('Round %d',r);
accuracy(1,r) = ((tp + tn)/length(probs))*100
end
fprintf('\nThe Accuracy of logistic regression for Breast Cancer data is: %f Percentage \n', sum(accuracy)/r);