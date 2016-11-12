load('../../data/image_classfn/CompleteData.mat');
numClasses = 5;

trainData = [];
trainLabel = [];
valData = [];
valLabel = [];
testData = [];
testLabel = [];


for i = 1:numClasses
   switch i
       case 1
           tempData = CompleteData{17,1};
       case 2
           tempData = CompleteData{8,1};
       case 3
           tempData = CompleteData{19,1};
       case 4
           tempData = CompleteData{10,1};
       case 5
           tempData = CompleteData{6,1};
   end
   tempSize = size(tempData,1);
   [trainTemp, valTemp, testTemp] = dividerand(tempSize, 0.7,0.15,0.15);
   for j = 1:size(trainTemp,2)
        trainData = [trainData ; tempData(trainTemp(1,j),:)];
        trainLabel = [trainLabel ; i];
   end
   for j = 1:size(valTemp,2)
        valData = [valData ; tempData(valTemp(1,j),:)];
        valLabel = [valLabel ; i];
   end
   for j = 1:size(testTemp,2)
        testData = [testData ; tempData(testTemp(1,j),:)];
        testLabel = [testLabel ; i];
   end
   
end

model = svmtrain(trainLabel, trainData,'-s 0 -t 1 -q -d 2');
output_labels_val = svmpredict(valLabel, valData, model);
output_labels_test = svmpredict(testLabel, testData, model);