% x = [1 2 3 4 5];
% y=[59.73 88.59 89.93 44.30 20.13];

x = [10^-5 10^-4 10^-3 10^-2 10^-1 1 10^1 10^2 10^3 10^4 10^5];
y=[88.59 ];

plot(x,y);
hold on;
title('Accuracy vs degree plot');
ylabel('Accuracy(in %)');
xlabel('degree');