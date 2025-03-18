function [sum] = jiec(num)%返回10的多少次方
sum=0;
for i=1:num
    sum=sum+log10(i);
end
end
