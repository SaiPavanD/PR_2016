root = '../data/non-linearly_Separable/';
numClasses = 3;
numHiddenNeurons = [3 3 3];
step = 0.01;

inputs = [];
targets = [];

for i = 1:numClasses
   temp_path = 'class';
   temp_path = strcat(root,temp_path,int2str(i),'_train.txt');
   temp_data = dlmread(temp_path)';
   inputs = [inputs temp_data];
   temp_targets = zeros(numClasses,size(temp_data,2));
   temp_targets(i,:) = temp_targets(i,:) + 1;
   targets = [targets temp_targets];
end

[~,trainInd] = size(inputs);
[~,trainClassfn] = max(targets);

for i = 1:numClasses
   temp_path = 'class';
   temp_path = strcat(root,temp_path,int2str(i),'_val.txt');
   temp_data = dlmread(temp_path)';
   inputs = [inputs temp_data];
   temp_targets = zeros(numClasses,size(temp_data,2));
   temp_targets(i,:) = temp_targets(i,:) + 1;
   targets = [targets temp_targets];
end

[~,valInd] = size(inputs);

for i = 1:numClasses
   temp_path = 'class';
   temp_path = strcat(root,temp_path,int2str(i),'_test.txt');
   temp_data = dlmread(temp_path)';
   inputs = [inputs temp_data];
   temp_targets = zeros(numClasses,size(temp_data,2));
   temp_targets(i,:) = temp_targets(i,:) + 1;
   targets = [targets temp_targets];
end

[~,testInd] = size(inputs);

net = patternnet(numHiddenNeurons);
net.trainParam.epochs=10000;
net.trainParam.max_fail=5000;
net.trainParam.goal=1e-8;
net.trainParam.min_grad=1e-8;
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:trainInd;
net.divideParam.valInd = trainInd+1:valInd;
net.divideParam.testInd = valInd+1:testInd;
[net,tr] = train(net,inputs,targets);

[x,y] = meshgrid(min(inputs(1,:)):step:max(inputs(1,:)),min(inputs(2,:)):step:max(inputs(2,:)));
decisionRegionPoints = [x(:)';y(:)'];
decisionRegionOutputs = net(decisionRegionPoints);
[~,decisionRegionClassfn] = max(decisionRegionOutputs);

colors=[1 0.75 0.75; 0.75 1 0.75; 0.75 0.75 1];
plot1 = gscatter(decisionRegionPoints(1,:),decisionRegionPoints(2,:),decisionRegionClassfn,colors,'***',[],'off');
hold on;
xlabel('Dimension 1');
ylabel('Dimension 2');
plot2 = gscatter(inputs(1,1:trainInd),inputs(2,1:trainInd),trainClassfn,'rgb','...');
hold on;
legend([plot2],'Class 1','Class 2','Class 3');
title('Non-linearly seperable data using MLFFNN');