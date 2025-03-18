function [Hnei] = neinei(S,n_inliers)
bigbind=ceil(quantile(sum(S),0.9));
if bigbind<n_inliers
    bigbind=n_inliers;
end
N=length(S);
duan=floor(N/100);Hnei=[];
parfor i=1:100
    first=duan*(i-1)+1;
    if i==100
        mo=N;
    else
        mo=duan*i;
    end
    [~,HH] = maxk(S(first:mo,:),bigbind,2);
    Hnei=[Hnei;HH];
end
end

