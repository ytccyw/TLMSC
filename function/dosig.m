function [sig] = dosig(inputs)
A=2;B=7;
peclu=160;
[repClsLabel,centerpoint] = litekmeans(inputs,peclu,'MaxIter',20);
CD=dis(   centerpoint,centerpoint  );
[~,cindex] = mink(CD,2,2);
flag=0;
sig=zeros(size(inputs,1),1);
for s=1:peclu
    temp=repClsLabel==s;
    temp1=logical(sum(repClsLabel==cindex(s,:),2));
    Xclu=inputs(temp,:);
    Xclu1=inputs(temp1,:);
    if size(Xclu1,1)>B-1
        zhi=sqrt(  dis(   Xclu,Xclu1  )   );
        [forsig,~] = mink(zhi,B,2);
        sig(temp,1)=max(mean(forsig(:,A:B),2),1e-10);
    else
        flag=1;
        sig(temp,1)=1;
    end
end
if flag==1
sig(sig==1)=mean(sig(sig~=1));
end
end

