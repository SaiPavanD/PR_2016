classdef NaiveBayesClassifier
    properties
        train
        test
        val
        numTest
        numVal
        n
        m
        c
        p
    end
    
    methods
        
        function result = mv(obj,test,mean,sig)
            d = size(test, 2);
            diff = test - mean;
            sig=sig+10^-5 * eye(d);
            s=abs(det(sig));
            result=s^-0.5 * exp(-0.5*diff * pinv(sig)*diff');
        end
        
        
        function obj=setParam(obj)
           sum=0;
           for i=1:obj.n
               obj.m{i} = mean(obj.train{i});
               obj.c{i} = diag(diag(cov(obj.train{i})));
               num{i}=size(obj.train{i},1);
               sum = sum+num{i};
               
           end
           
           for i=1:obj.n
                obj.p{i}=num{i}/sum;
           end
        end
        
        
        function class = classify(obj,test)
            f=[];
            
            for i=1:obj.n
                temp = mv(obj,test,obj.m{i},obj.c{i})*obj.p{i};
                f=[f, temp];
            end
                     
            [~, class]=max(f);
        end
        
        function accu = validate(obj)
            num = size(obj.val,1);
            
            accu=0;
            for i = 1:num
               result=classify(obj,obj.val(i,1:48));
               actual=obj.val(i,49);
               if actual==result
                   accu=accu+1;
               end
            end
            accu=accu/num;
        end
        
        function conf = getAccu(obj)
            num = size(obj.test,1);
            
            
            conf=zeros(obj.n,obj.n);
            for i = 1:num
               result=classify(obj,obj.test(i,1:48));
               actual=obj.test(i,49);
               conf(actual,result)=conf(actual,result)+1;
            end
            
            for i=1:obj.n
                conf(i,1:obj.n)=conf(i,1:obj.n)/obj.numTest(i,1);
            end
            
        end
        
        
    end
end
