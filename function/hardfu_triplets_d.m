function[d]= hardfu_triplets_d(p,class,cnum)
d=1;
while true    
    gailv=0;
    for i=0:d-1
        mi=(jiec(cnum)+jiec(cnum*(class-1))+jiec(p)+jiec(class*cnum-p)-jiec(cnum-i)-jiec(p-i)-jiec(class*cnum)-jiec(class*cnum-cnum-p+i));
        gailv =gailv+(10^mi)/factorial(i);
    end
    if gailv>0.97
        break
    end
    d=d+1;
end
end