function [S] = nnxiangsixing(inputs,n_inliers,sig)
N=size(inputs,1);
NW=ceil(N/1000);
if sig~=0
    zzz=[];lll=[];
    parfor half=1:NW
        first=1000*(half-1)+1;
        if half==NW
            mo=N;
        else
            mo=1000*half;
        end
        D = dis(inputs(first:mo,:),inputs);
        D=exp(-D./(sig(first:mo).*sig'));
        [zi,li] = maxk(D,n_inliers,2);
        zzz=[zzz;zi];lll=[lll;li];
    end
    hang=repmat((1:N)',1,n_inliers);
    SS = sparse(hang(:),lll(:),zzz(:));
else
    clear sig
    A=5;B=7;
    D=pdist2(inputs, inputs);
    [forsig,~] = mink(D,B,2);
    sig=max(mean(forsig(:,A:B),2),1e-10);
    D=exp(-(D.^2)./(sig.*sig'));
    [zhi,lie] = maxk(D,n_inliers,2);
    hang=repmat((1:N)',1,n_inliers);
    SS = sparse(hang(:),lie(:),zhi(:));
end
S=SS*SS;
end