function [wei,tri] = triplet_weights(tri,S)
N=length(tri);
wei=zeros(N,1);
for i=1:N
    wei(i,1)=    (S(tri(i,1),tri(i,2))-S(tri(i,1),tri(i,3)));
    if wei(i,1)<0       
        temp=tri(i,1);
        tri(i,1)=tri(i,2);
        tri(i,2)=temp;
        wei(i,1)=abs(wei(i,1));
    end
end
end

