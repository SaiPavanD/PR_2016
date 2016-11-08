load('../data/image_classfn/CompleteData.mat');
numClasses = 5;
numHiddenLayers = 10;
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
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;
[net,tr] = train(net,inputs,targets);

% [x,y] = meshgrid(min(inputs(1,:)):step:max(inputs(1,:)),min(inputs(2,:)):step:max(inputs(2,:)));
% decisionRegionPoints = [x(:)';y(:)'];
% decisionRegionOutputs = net(decisionRegionPoints);
% [~,decisionRegionClassfn] = max(decisionRegionOutputs);
% 
% colors=[1 0.75 0.75; 0.75 1 0.75; 0.75 0.75 1; 1 1 0.75];
% plot1 = gscatter(decisionRegionPoints(1,:),decisionRegionPoints(2,:),decisionRegionClassfn,colors,'****',[],'off');
% hold on;
% xlabel('Dimension 1');
% ylabel('Dimension 2');
% plot2 = gscatter(inputs(1,1:trainInd),inputs(2,1:trainInd),trainClassfn,'rgby','....');
% hold on;
% legend([plot2],'Class 1','Class 2','Class 3','Class 4');
% title('Overlapping data using MLFFNN');