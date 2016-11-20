load('../data/image_classfn/CompleteData.mat');
numClasses = 5;
cond = 1;

data{1}=CompleteData{17,1};
data{2}=CompleteData{8,1};
data{3}=CompleteData{19,1};
data{4}=CompleteData{10,1};
data{5}=CompleteData{6,1};
totalImageData = [data{1,1}; data{1,2}; data{1,3}; data{1,4}; data{1,5}];
Z = getPCA(totalImageData,993);
Z1 = Z(1:216,:);
Z2 = Z(217:371,:);
Z3 = Z(372:562,:);
Z4 = Z(563:764,:);
Z5 = Z(765:993,:);


while cond == 1
    trainData = [];
    trainLabel = [];
    valData = [];
    valLabel = [];
    testData = [];
    testLabel = [];
    for i = 1:numClasses
       switch i
           case 1
               tempData = Z1;
           case 2
               tempData = Z2;
           case 3
               tempData = Z3;
           case 4
               tempData = Z4;
           case 5
               tempData = Z5;
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

    model = svmtrain(trainLabel, trainData,'-s 0 -c 10 -t 2 -q -g 1e-10');
    [output_labels_val, accu, ~] = svmpredict(valLabel, valData, model);
    accu(1,1)
    if accu(1,1)>75
        cond = 0;
    end
end

output_labels_test = svmpredict(testLabel, testData, model);

testTargets = zeros(numClasses,size(testData,1));
testResults = zeros(numClasses,size(testData,1));
for i=1:size(testData,1)
    testTargets(testLabel(i,1),i)=1;
    testResults(output_labels_test(i,1),i)=1;
end
plotconfusion(testTargets, testResults);