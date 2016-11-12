root = '../data/linearly_Separable_Data/';
numClasses = 4;
c = 1;
step = 0.05;

train_data = [];
train_labels = [];
test_data = [];
test_labels = [];

for i = 1:numClasses
   temp_path = 'class';
   temp_path = strcat(root,temp_path,int2str(i),'_train.txt');
   temp_data = dlmread(temp_path);
   train_data = [train_data ; temp_data];
   temp_labels = zeros(size(temp_data,1),1);
   temp_labels(:,1) = i;
   train_labels = [train_labels ; temp_labels];
end

% for i = 1:numClasses
%    temp_path = 'class';
%    temp_path = strcat(root,temp_path,int2str(i),'_val.txt');
%    temp_data = dlmread(temp_path);
%    train_data = [train_data ; temp_data];
%    temp_labels = zeros(size(temp_data,1),1);
%    temp_labels(:,1) = i;
%    train_labels = [train_labels temp_labels];
% end

for i = 1:numClasses
   temp_path = 'class';
   temp_path = strcat(root,temp_path,int2str(i),'_test.txt');
   temp_data = dlmread(temp_path);
   test_data = [test_data ; temp_data];
   temp_labels = zeros(size(temp_data,1),1);
   temp_labels(:,1) = i;
   test_labels = [test_labels ; temp_labels];
end

model = svmtrain(train_labels, train_data,'-s 0 -t 0 -q -c 2');
% output_labels = svmpredict(test_labels, test_data, model);

[x,y] = meshgrid(-20:step:20,-20:step:20);
decisionRegionPoints = [x(:),y(:)];
dummyLabels = zeros(size(decisionRegionPoints,1),1);
decisionRegionOutputs = svmpredict(dummyLabels, decisionRegionPoints, model,'-q');

colors=[1 0.75 0.75; 0.75 1 0.75; 0.75 0.75 1; 1 1 0.75];
plot1 = gscatter(decisionRegionPoints(:,1),decisionRegionPoints(:,2),decisionRegionOutputs,colors,'****',[],'off');
hold on;
xlabel('Dimension 1');
ylabel('Dimension 2');
plot2 = gscatter(train_data(:,1),train_data(:,2),train_labels,'rgby','....');
hold on;
legend([plot2],'Class 1','Class 2','Class 3','Class 4');
title('Linearly seperable data using MLFFNN');