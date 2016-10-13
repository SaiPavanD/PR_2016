load('../../Dataset_Assignment1/Dataset-2_real_world/a_Image Classification data/CompleteData.mat');

data{1}=CompleteData{17,1};
data{2}=CompleteData{8,1};
data{3}=CompleteData{19,1};
data{4}=CompleteData{10,1};
data{5}=CompleteData{6,1};

%standardize
for i=1:5
   for j=1:size(data{i},2)
      m = mean(data{i}(:,j));
      v = sqrt(var(data{i}(:,j)));
      data{i}(:,j)=(data{i}(:,j)-m)/v;
   end
end


a = NaiveBayesClassifier;
a.n=5;
acc=0;

while acc<0.2  %threshold
    for i=1:5
        rand{i} = randperm(size(data{i},1)); 
    end
    
    a.val=[];
    a.test=[];
    for i=1:5
           a.train{i}=[];
           

        for j=rand{i}(1,1:floor(0.7*size(data{i},1)))
            a.train{i}=[a.train{i};data{i}(j,:)];
        end
        for j=rand{i}(1,ceil(0.7*size(data{i},1)):floor(0.8*size(data{i},1)))
           a.val=[a.val;data{i}(j,:),i];
        end
        tempnumtest=0;
        for j=rand{i}(1,ceil(0.85*size(data{i},1)):size(data{i},1))
            a.test=[a.test;data{i}(j,:), i];
            tempnumtest=tempnumtest+1;
        end
        a.numTest=[a.numTest;tempnumtest];
    end
    
    a=setParam(a);
    acc = validate(a)
end
a=setParam(a);

confusion=getAccu(a)