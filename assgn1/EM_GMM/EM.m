classdef EM
    properties
        train
        n
        total_train=0;
        prior
        q
        clusters
        w_old
        m_old
        c_old
        class_wise
        conf = zeros(10,10)
    end
    
    methods
        function  obj = set_train(obj,class,files)
            path = 'Team19\';
            path = strcat(path,class);
            path = strcat(path,'\Train\');
            traindata = [];
            
            for i=1:size(files,2)
               temp_path = strcat(path,files{i});
               temp = dlmread(temp_path);
               temp = temp(2:end,:);
               traindata = [traindata;temp];
            end
            
            traindata = standardise(obj,traindata);
            obj.train = traindata;
            obj.n = size(traindata,1);
            obj.total_train = obj.total_train + obj.n;
        end
        
        function inp = standardise(obj,inp)
            for i=1:size(inp,2)
                m = mean(inp(:,i));
                v = sqrt(var(inp(:,i)));
                inp(:,i)=(inp(:,i)-m)/v; 
            end
        end
        
        function obj = K_means(obj)
            idx = kmeans(obj.train,obj.q);
            obj.clusters{obj.q}=[];
            obj.w_old{obj.q}=[];
            obj.m_old{obj.q}=[];
            obj.c_old{obj.q}=[];
            for i=1:size(idx,1)
               obj.clusters{idx(i,1)}=[obj.clusters{idx(i,1)} ; obj.train(i,:)];
            end
            for i=1:obj.q
                obj.w_old{i}=size(obj.clusters{i},1)/obj.n;
                obj.m_old{i}=mean(obj.clusters{i});
%                 size(cov(obj.clusters{i}))
%                 size(eye(size(obj.clusters{i},1)))
                obj.c_old{i}=cov(obj.clusters{i});
                
            end
        end
        
        function obj = get_new(obj)
            
            sum = zeros(obj.n,1);
            for i = 1:obj.n
                for j = 1:obj.q
                    %size(obj.train(i,:))
                    %size(obj.m_old{j})
%                     j
%                     size(obj.c_old{j})
                    if det(obj.c_old{j})==0
%                         obj.c_old{j}
                        obj.c_old{j}=obj.c_old{j}+(10^-1)*eye(size(obj.clusters{j},2));
%                         det(obj.c_old{j})
                    end
                   
                    
%                     det(obj.c_old{j})
                   sum(i,1)=sum(i,1)+obj.w_old{j}*mvnpdf(obj.train(i,:),obj.m_old{j},obj.c_old{j});
                end
            end
            
            r = zeros(obj.n,obj.q);
            for i = 1:obj.n
                for j = 1:obj.q
                    if det(obj.c_old{j})==0
%                         obj.c_old{j}
                        obj.c_old{j}=obj.c_old{j}+(10^-1)*eye(size(obj.clusters{j},2));
%                         det(obj.c_old{j})
                    end
                   r(i,j)=obj.w_old{j}*mvnpdf(obj.train(i,:),obj.m_old{j},obj.c_old{j})/sum(i,1);
                end
            end
            
            w{obj.q}=[];
            for i=1:obj.q
                w{i}=0;
               for j=1:obj.n
                   w{i}=w{i}+r(j,i);
               end
               w{i}=w{i}/obj.n;
            end
            
%             for i=1:obj.q
%                 for j=1:obj.n
%                     lambda(i,1)=lambda(i,1)+r(j,i)/obj.w_old{i};
%                 end
%             end
%             w{obj.q}=[];
%             for i=1:obj.q
%                w{i}= size(obj.clusters{i},1)/lambda(i,1);
%             end
            
            m{obj.q}=[];
            for i=1:obj.q
                m{i}=0;
               for j=1:obj.n
                   m{i}=m{i}+r(j,i)*obj.train(j,:);
               end
               m{i}=m{i}/size(obj.clusters{i},1);
            end
            
            c{obj.q}=[];
            for i=1:obj.q
                c{i}=zeros(size(obj.c_old,1));
               for j=1:obj.n
                   diff = obj.train(j,:)-m{i};
                   c{i}=c{i}+r(j,i)*(diff'*diff);
               end
               c{i}=c{i}/size(obj.clusters{i},1);
            end
            
            obj.w_old=w;
            obj.m_old=m;
            obj.c_old=c;
            
        end
        
        function obj = iterate(obj,i)   %save params for i'th class
            for j=1:10
               obj=get_new(obj);
            end
            obj.class_wise{i}={};
            obj.class_wise{i}{1}=obj.w_old;
            obj.class_wise{i}{2}=obj.m_old;
            obj.class_wise{i}{3}=obj.c_old;
            obj.class_wise{i}{4}=obj.q;
        end
        
        function obj = set_prior_no(obj,i)   %training data of that i'th class
            obj.prior{i}=obj.n;
        end
        
        function obj = set_prior(obj)
            for i=1:size(obj.prior,2)
                obj.prior{i}=obj.prior{i}/obj.total_train;
            end
        end
        
        function result=classify(obj,test,c)    %c= num of classes
            list = zeros(c,1);
            
            for i=1:c
                for j=1:size(test,1)
                    temp=0;
                    for k=1:obj.class_wise{i}{4}
                        if det(obj.class_wise{i}{3}{k})==0
                            obj.class_wise{i}{3}{k}=obj.class_wise{i}{3}{k}+(10^-4)*eye(39);
                        end
                        temp=temp+obj.class_wise{i}{1}{k}*mvnpdf(test(j,:),obj.class_wise{i}{2}{k},obj.class_wise{i}{3}{k});
                    end
                    list(i,1)=list(i,1)+log(temp);
                end
                list(i,1)=list(i,1)+log(obj.prior{i});
            end
            
            [~,result] = max(list);
           
        end
        
        function obj = get_conf(obj,class_num,num_classes,class_name,test_names)
                path = 'Team19\';
                path = strcat(path,class_name);
                path = strcat(path,'\Test\');
                
                for i = 1 : size(test_names,2)
                    temp_path = strcat(path,test_names{i});
                    data = dlmread(temp_path);
                    data = data(2:end,:);
                    
                    res = classify(obj,data,num_classes);
                    obj.conf(class_num,res)=obj.conf(class_num,res)+1;
                end
                
                obj.conf(class_num,:)=obj.conf(class_num,:)/10;
                
            end
        
    end
end