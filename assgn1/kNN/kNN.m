classdef kNN
    properties
        trainData
        valData
        testData
        rangeK
        optK
        step %step size
%         numTrain
%         numVal
%         numTest
    end
    methods
        function class = getClass(obj,test)
            [numRows , ~] = size(obj.trainData);
            
            dist=zeros(numRows,2);
            for i = 1:numRows
                dist(i,1) = (obj.trainData(i,1)-test(1,1))^2+(obj.trainData(i,2)-test(1,2))^2;
                dist(i,2) =  obj.trainData(i,3);
            end
            
            distSorted = sortrows(dist,1);

            %obj.rangeK=ceil(sqrt(numRows));    %range of k
            class=zeros(1,obj.rangeK);
            for i = 1:obj.rangeK    
                class(1,i) = mode(distSorted(1:i,2));
            end
        end
        
        function class = getClassGivenK(obj,test,k)
            [numRows , ~] = size(obj.trainData);
            
            dist=zeros(numRows,2);
            for i = 1:numRows
                dist(i,1) = (obj.trainData(i,1)-test(1,1))^2+(obj.trainData(i,2)-test(1,2))^2;
                dist(i,2) =  obj.trainData(i,3);
            end
            
            distSorted = sortrows(dist,1);

            %obj.rangeK=ceil(sqrt(numRows));    %range of k
            class = mode(distSorted(1:k,2));
        end
        
        function accu_val = validate(obj)
            [numRows,~]=size(obj.valData);
            [numRowsTrain,~]=size(obj.trainData);
            
            obj.rangeK=ceil(sqrt(numRowsTrain));
            accu_val=zeros(1,obj.rangeK);
           
            for i =1:numRows
                tempClassi= getClass(obj,obj.valData(i,1:2));
                for j=1:obj.rangeK
                   if tempClassi(1,j)==obj.valData(i,3)
                        accu_val(1,j)=accu_val(1,j)+1;
                   end
                end
            end
            
            accu_val = accu_val / numRows
            
        end
        
        function accu_test = testKNN(obj)
            [numRows,~]=size(obj.testData);
            [numRowsTrain,~]=size(obj.trainData);
            
            obj.rangeK=ceil(sqrt(numRowsTrain));
            accu_test=zeros(1,obj.rangeK);
           
            for i =1:numRows
                tempClassi= getClass(obj,obj.testData(i,1:2));
                for j=1:obj.rangeK
                   if tempClassi(1,j)==obj.testData(i,3)
                        accu_test(1,j)=accu_test(1,j)+1;
                   end
                end
            end
            
            accu_test = accu_test / numRows   
        end
        
        function plotDecisionRegion(obj)
            x2 = max(obj.valData(:,1));
            x1 = min(obj.valData(:,1));
            y2 = max(obj.valData(:,2));
            y1 = min(obj.valData(:,2));

            

            [x,y]=meshgrid(x1:obj.step:x2,y1:obj.step:y2);
            points=[x(:),y(:)];

            numPoints=size(points,1);
            points=[points zeros(numPoints,1)];
            
            k=21;       %k value
            for i=1:numPoints
               points(i,3)= getClassGivenK(obj,points(i,1:2),k);
            end
            
            colors=[1 0.75 0.75; 0.75 1 0.75; 0.75 0.75 1; 1 1 0.75];
            
            
            gscatter(points(:,1),points(:,2),points(:,3),colors,'****',[],'off');
            
            hold on;
            
            xlabel('Dimension 1');
            ylabel('Dimension 2');
        end
    end
end