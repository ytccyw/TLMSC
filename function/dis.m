function D = dis(X,Y)
Yt = Y';
D = abs(bsxfun(@plus,sum(X.*X,2),sum(Yt.*Yt,1))-2*X*Yt);
end
