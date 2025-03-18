function groups = fastSpectralClustering(em,n,numflag)
warning off;
N = size(em,1);
num=floor(N^0.5);
REPlic = 20;
options = statset('UseParallel',0);
if N>5000
    REPlic = 6;
end

if numflag==1
hang=repmat((1:N)',1,num)';
[zhi,iii]=pdist2(em,em, 'squaredeuclidean','Smallest',num);
zhi=1./(1+zhi);
S = sparse(hang(:),iii(:),zhi(:));

zhi = 1./sqrt(sum(S)+eps);
index=1:length(S);
DN=sparse(index,index,zhi);

L=DN * S * DN;
[~,~, kerN] = rSVDBKI(L,n,4);
kerNS = kerN ./ (sqrt(sum(kerN.^2, 2)) + 1e-10);
groups = kmeans(kerNS,n,'replicates',REPlic,'Options',options);

else
groups = kmeans(em,n,'replicates',REPlic,'Options',options);
end