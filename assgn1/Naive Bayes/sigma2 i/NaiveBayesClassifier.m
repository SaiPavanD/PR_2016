classdef NaiveBayesClassifier
    properties
        train1
        train2
        train3
        train4
        trainData
        test
        valData
        numTest
        step
    end
    
    methods
        function class = classify4(obj,test)
            m1 = mean(obj.train1);
            m2 = mean(obj.train2);
            m3 = mean(obj.train3);
            m4 = mean(obj.train4);
            
            c = diag(diag(cov(obj.trainData(:,1:2))));
            c = sqrt(det(c))*eye(2);
            
            l1=mvnpdf(test,m1,c);
            l2=mvnpdf(test,m2,c);
            l3=mvnpdf(test,m3,c);
            l4=mvnpdf(test,m4,c);
            
            num1=size(obj.train1,1);
            num2=size(obj.train2,1);
            num3=size(obj.train3,1);
            num4=size(obj.train4,1);
            
            total=num1+num2+num3+num4;
            
            p1=num1/total;
            p2=num2/total;
            p3=num3/total;
            p4=num4/total;
            
            [~, class]=max([l1*p1,l2*p2,l3*p3,l4*p4]);
        end
        
         function class = classify3(obj,test)
            m1 = mean(obj.train1);
            m2 = mean(obj.train2);
            m3 = mean(obj.train3);
%             m4 = mean(obj.train4);
            
            c = diag(diag(cov(obj.trainData(:,1:2))));
            c = sqrt(det(c))*eye(2);
            
            l1=mvnpdf(test,m1,c);
            l2=mvnpdf(test,m2,c);
            l3=mvnpdf(test,m3,c);
%             l4=mvnpdf(test,m4,c);
            
            num1=size(obj.train1,1);
            num2=size(obj.train2,1);
            num3=size(obj.train3,1);
%             num4=size(obj.train4,1);
            
            total=num1+num2+num3;
            
            p1=num1/total;
            p2=num2/total;
            p3=num3/total;
%             p4=num4/total;
            
            [~, class]=max([l1*p1,l2*p2,l3*p3]);
        end
        
        function conf = getAccu(obj)
            num = size(obj.test,1);
            
            n = size(obj.numTest,1);      %num classes
            conf=zeros(n,n);
            for i = 1:num
               result=classify4(obj,obj.test(i,1:2));
               actual=obj.test(i,3);
               conf(actual,result)=conf(actual,result)+1;
            end
            
            for i=1:n
                conf(i,1:n)=conf(i,1:n)/obj.numTest(i,1);
            end
            
        end
        
        function conf = getAccu3(obj)
            num = size(obj.test,1);
            
            n = size(obj.numTest,1);      %num classes
            conf=zeros(n,n);
            for i = 1:num
               result=classify3(obj,obj.test(i,1:2));
               actual=obj.test(i,3);
               conf(actual,result)=conf(actual,result)+1;
            end
            
            for i=1:n
                conf(i,1:n)=conf(i,1:n)/obj.numTest(i,1);
            end
            
        end
        
         function plotDecisionRegion4(obj)
            x2 = max(obj.valData(:,1));
            x1 = min(obj.valData(:,1));
            y2 = max(obj.valData(:,2));
            y1 = min(obj.valData(:,2));

            

            [x,y]=meshgrid(x1:obj.step:x2,y1:obj.step:y2);
            points=[x(:),y(:)];

            numPoints=size(points,1);
            points=[points zeros(numPoints,1)];
            
           
            for i=1:numPoints
               points(i,3)= classify4(obj,points(i,1:2));
            end
            
            colors=[1 0.75 0.75; 0.75 1 0.75; 0.75 0.75 1; 1 1 0.75];
            
            
            gscatter(points(:,1),points(:,2),points(:,3),colors,'****',[],'off');
            
            hold on;
            
            xlabel('Dimension 1');
            ylabel('Dimension 2');
         end
         
         function plotDecisionRegion3(obj)
            x2 = max(obj.valData(:,1));
            x1 = min(obj.valData(:,1));
            y2 = max(obj.valData(:,2));
            y1 = min(obj.valData(:,2));

            

            [x,y]=meshgrid(x1:obj.step:x2,y1:obj.step:y2);
            points=[x(:),y(:)];

            numPoints=size(points,1);
            points=[points zeros(numPoints,1)];
            
           
            for i=1:numPoints
               points(i,3)= classify3(obj,points(i,1:2));
            end
            
            colors=[0.75 0.75 1; 0.75 1 0.75; 0.75 0.75 1; 1 1 0.75];
            
           
            gscatter(points(:,1),points(:,2),points(:,3),colors,'****',[],'off');
            
            hold on;
            
            xlabel('Dimension 1');
            ylabel('Dimension 2');
         end
        
    end
end
