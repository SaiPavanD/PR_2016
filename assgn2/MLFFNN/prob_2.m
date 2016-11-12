load('../data/image_classfn/CompleteData.mat');
numClasses = 5;
numHiddenLayers = 5;
% step = 0.05;

inputs = [];
targets = [];

for i = 1:numClasses
   switch i
       case 1
           temp_data = CompleteData{17,1}';
       case 2
           temp_data = CompleteData{8,1}';
       case 3
           temp_data = CompleteData{19,1}';
       case 4
           temp_data = CompleteData{10,1}';
       case 5
           temp_data = CompleteData{6,1}';
   end
   for j = 1:size(temp_data,1)
        temp_data(j,:) = (temp_data(j,:) - mean(temp_data(j,:)))/var(temp_data(j,:));
   end
   inputs = [inputs temp_data];
   temp_targets = zeros(numClasses,size(temp_data,2));
   temp_targets(i,:) = temp_targets(i,:) + 1;
   targets = [targets temp_targets];
end

net = patternnet(numHiddenLayers);
net.trainParam.epochs=100000;
net.trainParam.max_fail=50000;
net.trainParam.goal=1e-8;
net.trainParam.min_grad=1e-8;
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.75;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;
[net,tr] = train(net,inputs,targets);