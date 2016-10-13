a = NaiveBayesClassifier;

class1_train = dlmread('../../Dataset_Assignment1/Dataset-1_2Dimensional/non-linearly_Separable/group19/class1_train.txt');
class2_train = dlmread('../../Dataset_Assignment1/Dataset-1_2Dimensional/non-linearly_Separable/group19/class2_train.txt');
class3_train = dlmread('../../Dataset_Assignment1/Dataset-1_2Dimensional/non-linearly_Separable/group19/class3_train.txt');
% class4_train = dlmread('../Dataset_Assignment1/Dataset-1_2Dimensional/overlapping_data/group19/class4_train.txt');

a.train1=class1_train;
a.train2=class2_train;
a.train3=class3_train;
% a.train4=class4_train;

num=size(class1_train,1);
class = zeros(num,1);
class = class + 1;
class1_train = [class1_train class];
num=size(class2_train,1);
class = zeros(num,1);
class = class + 2;
class2_train = [class2_train class];
num=size(class3_train,1);
class = zeros(num,1);
class = class + 3;
class3_train = [class3_train class];
% num=size(class4_train,1);
% class = zeros(num,1);
% class = class + 4;
% class4_train = [class4_train class];


a.trainData = [class1_train; class2_train; class3_train];




%gscatter(a.trainData(:,1),a.trainData(:,2),a.trainData(:,3),'rgby','....');
% plot(class1_train(:,1),class1_train(:,2),'r.');    %class1 - red
% hold on;
% plot(class2_train(:,1),class2_train(:,2),'g.');    %class2 - green
% hold on;
% plot(class3_train(:,1),class3_train(:,2),'b.');    %class3 - blue
% hold on;
% plot(class4_train(:,1),class4_train(:,2),'y.');    %class4 - yellow
% hold on;
% plot(test(1),test(2),'k*');

class1_val = dlmread('../../Dataset_Assignment1/Dataset-1_2Dimensional/non-linearly_Separable/group19/class1_val.txt');
class2_val = dlmread('../../Dataset_Assignment1/Dataset-1_2Dimensional/non-linearly_Separable/group19/class2_val.txt');
class3_val = dlmread('../../Dataset_Assignment1/Dataset-1_2Dimensional/non-linearly_Separable/group19/class3_val.txt');
% class4_val = dlmread('../Dataset_Assignment1/Dataset-1_2Dimensional/linearly_Separable_Data/group19/class4_val.txt');
a.valData = [class1_val; class2_val; class3_val];


%test
class1_val = dlmread('../../Dataset_Assignment1/Dataset-1_2Dimensional/non-linearly_Separable/group19/class1_test.txt');
class2_val = dlmread('../../Dataset_Assignment1/Dataset-1_2Dimensional/non-linearly_Separable/group19/class2_test.txt');
class3_val = dlmread('../../Dataset_Assignment1/Dataset-1_2Dimensional/non-linearly_Separable/group19/class3_test.txt');
% class4_val = dlmread('../Dataset_Assignment1/Dataset-1_2Dimensional/overlapping_data/group19/class4_test.txt');
num=size(class1_val,1);
a.numTest = [num];
class = zeros(num,1);
class = class + 1;
class1_val = [class1_val class];
num=size(class2_val,1);
a.numTest = [a.numTest; num];
class = zeros(num,1);
class = class + 2;
class2_val = [class2_val class];
num=size(class3_val,1);
a.numTest = [a.numTest; num];
class = zeros(num,1);
class = class + 3;
class3_val = [class3_val class];
% num=size(class4_val,1);
% a.numTest = [a.numTest; num];
% class = zeros(num,1);
% class = class + 4;
% class4_val = [class4_val class];

a.test = [class1_val; class2_val; class3_val];

getAccu3(a)


a.step=0.1;
plotDecisionRegion3(a);

p1 = gscatter(a.trainData(:,1),a.trainData(:,2),a.trainData(:,3),'rgb','...');
hold on;
legend([p1],'Class 1','Class 2','Class 3');
title('Non-linearly seperable data using Naive Bayes Classifier');