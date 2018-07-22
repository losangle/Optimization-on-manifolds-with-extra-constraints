function paretoplot
    A = dlmread('zz_NonnegPCAParetoDim10Repeat100Exp1.dat',  ',');
    [nrow, ncol] = size(A);
    A(A==0) = eps;
    minimax = A(:, 1:2);
    alm = A(:, 4:5);
    lqh = A(:, 7:8);
    lse = A(:, 10:11);
    fmincon = A(:, 13:14);
    
    mincost = min(min(A(:, [5 8 11 14])));
    alm(:, 2) = alm(:,2) - mincost;
    lqh(:, 2) = lqh(:,2) - mincost;
    lse(:, 2) = lse(:,2) - mincost;
    fmincon(:, 2) = fmincon(:,2) - mincost;
    
    figure
    hold on
    scatter(alm(:,1),alm(:,2), 25, 'r');
    scatter(lqh(:,1),lqh(:,2), 25, 'b');
    scatter(lse(:,1),lse(:,2), 25, 'g');
    scatter(fmincon(:,1),fmincon(:,2), 25, 'm');
    set(gca,'Xscale','log','Yscale','log')
    legend('alm','lqh', 'lse','fmincon');
    hold off
    
    figure
    hold on
    subplot(2,2,1);
    scatter(alm(:,1),alm(:,2), 25, 'r');
    xlabel('alm');
    subplot(2,2,2);
    scatter(lqh(:,1),lqh(:,2), 25, 'b');
    xlabel('lqh');
    subplot(2,2,3);
    scatter(lse(:,1),lse(:,2), 25, 'g');
    xlabel('lse');
    subplot(2,2,4);
    scatter(fmincon(:,1),fmincon(:,2), 25, 'm');
    xlabel('fmincon');
    hold off
end