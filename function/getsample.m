function [samples] = getsample(num, maxval, rejects)
samples=zeros(maxval,num);
for i=1:maxval
    new = randperm(maxval,num);
    Lia = ismember(new,rejects(i,:));
    list=find(Lia==1);
    if list
        for j=1:length(list)
            reject = true;
            while reject
                new1 = randperm(maxval,1);
                reject=false;
                if sum(find(rejects(i,:)==new1))||sum(find(new==new1))
                    reject=true;
                end
            end
            new(list(j))=new1;
        end
    end
    samples(i,:)=new;
end
end