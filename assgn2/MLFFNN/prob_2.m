load('../data/image_classfn/CompleteData.mat');
numClasses = 5;
hiddenNeurons = [30 30];
endIndex = 0;
tempSize = 0;
trainInd = [];
valInd = [];
testInd = [];

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
   tempSize = size(temp_data,2);
   [trainTemp, valTemp, testTemp] = dividerand(tempSize, 0.7,0.15,0.15);
   trainTemp = trainTemp + endIndex;
   valTemp = valTemp + endIndex;
   testTemp = testTemp + endIndex;
   endIndex = endIndex + tempSize;
   trainInd = [trainInd trainTemp];
   valInd = [valInd valTemp];
   testInd = [testInd testTemp];
   inputs = [inputs temp_data];
   temp_targets = zeros(numClasses,size(temp_data,2));
   temp_targets(i,:) = temp_targets(i,:) + 1;
   targets = [targets temp_targets];
end

net = patternnet(hiddenNeurons);
net.trainParam.epochs=10000;
net.trainParam.max_fail=5000;
net.trainParam.goal=1e-6;
net.trainParam.min_grad=1e-6;
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;
% net.layers{1}.transferFcn='tansig';
% net.layers{2}.transferFcn='tansig';
[net,tr] = train(net,inputs,targets);