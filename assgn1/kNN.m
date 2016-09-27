classdef kNN
    properties
        k
        trainData
    end
    methods
        function class = getClass(obj,test,k)
            [numRows , ~] = size(obj.trainData);
            dist=[];
            for i = 1:numRows
                dist = [dist; (obj.trainData(i,1)-test(1,1))^2+(obj.trainData(i,2)-test(1,2))^2     obj.trainData(i,3)];
            end
            sortrows(dist,1);
            class = mode(dist(1:k,2));
        end
    end
end

    
