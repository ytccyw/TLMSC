function groups = fastSpectralClustering(em,n,num)
warning off;
N = size(em,1);
REPlic = 20;
if N>5000
    REPlic = 6;
end

hang=repmat((1:N)',1,num)';
[zhi,iii]=pdist2(em,em, 'squaredeuclidean','Smallest',num);
zhi=1./(1+zhi);
S = sparse(hang(:),iii(:),zhi(:));

zhi = 1./sqrt(sum(S)+eps);
index=1:length(S);
DN=sparse(index,index,zhi);

L=DN * S * DN;
[~,~, kerN] = rSVDBKI(L,n,4);

parfor i = 1:N
    kerNS(i,:) = kerN(i,:) ./ norm(kerN(i,:)+eps);
end
options = statset('UseParallel',1);
groups = kmeans(kerNS,n,'replicates',REPlic,'Options',options);