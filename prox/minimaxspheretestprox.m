function minimaxsphertestprox

    d = 3;
    n = 10;
    % Create the problem structure.
    manifold = spherefactory(d);
    problem.M = manifold;
    M = problem.M;
    data = zeros(d,n);
    for pointnum = 1 : n
        data(:,pointnum) = problem.M.rand();  %not applicable to other manifolds
        data(1,:) = abs(data(1,:));
    end
    
    cost = @(X) costFun(data, X);
    grad = @(X) gradFun(data, X);
    regcost = @(X, Y) regcostFun (M, X, Y);
    reggradSphere = @(X, Y) reggradFunSphere(M, X, Y);
    problem.regcost = regcost;
    problem.reggrad = reggradSphere;
    
    % Define the problem cost function and its Euclidean gradient.
    problem.cost  = cost;
    problem.egrad = grad;
    
    %Set options
    options = [];
    
    checkgradient(problem);

    x0 = M.rand();
    [xCur, cost, stats, options] = rlbfgsprox(problem, x0, options);
    
    X = xCur;
    figure
    subplot(2,3,1);
    surfprofile(problem, X, M.randvec(X), M.randvec(X));
    subplot(2,3,2);
    surfprofile(problem, X, M.randvec(X), M.randvec(X));
    subplot(2,3,3);
    surfprofile(problem, X, M.randvec(X), M.randvec(X));
    subplot(2,3,4);
    surfprofile(problem, X, M.randvec(X), M.randvec(X));
    subplot(2,3,5);
    surfprofile(problem, X, M.randvec(X), M.randvec(X));
    subplot(2,3,6);
    surfprofile(problem, X, M.randvec(X), M.randvec(X));
    

    displayinfo(stats)

    
    figure
    surfprofile(problem, xCur);
    hold all
    plot3(0,0,getCost(problem, xCur), 'r.', 'MarkerSize', 25);
    hold off
    if d == 3
        figure;
        % Plot the sphere
        [sphere_x, sphere_y, sphere_z] = sphere(50);
        handle = surf(sphere_x, sphere_y, sphere_z);
        set(handle, 'FaceColor', [152,186,220]/255);
        set(handle, 'FaceAlpha', .5);
        set(handle, 'EdgeColor', [152,186,220]/255);
        set(handle, 'EdgeAlpha', .5);
        daspect([1 1 1]);
        box off;
        axis off;
        hold on;
        % Add the chosen points
        Y = cell2mat(xHistory);
        Y = 1.02*Y;
        [row, col] = size(Y);
        plot3(Y(1,:), Y(2,:), Y(3,:), 'r.', 'MarkerSize', 5);
        plot3(data(1,:), data(2,:), data(3,:), 'g.', 'MarkerSize', 20);
        % And connect the points which are at minimal distance,
%         % within some tolerance.
        for k = 1 : col-1
            i = k; j = k+1;
            plot3(Y(1, [i j]), Y(2, [i j]), Y(3, [i j]), 'k-');
        end
        hold off;
    end
    
    
    function val = regcostFun(M, X, Y)
        val = M.dist(X, Y)^2;
        val = val * 0.01;
    end

    function val = reggradFunOblique(M, X, Y)
        [n, m] = size(X);
        val = zeros(n, m);
        for col = 1: m
            xydiff = X(:, col)-Y(:, col);
            normdiff = norm(xydiff);
            if normdiff ~= 0
                valcol = xydiff/(normdiff*sqrt(1-normdiff^2/4));
                d = real(2*asin(.5*normdiff));
                val(:, col) = 2 * valcol * d;
            end
        end
        inners = sum(X.*val, 1);
        val = val - bsxfun(@times, X, inners);
    end
    
    function val = reggradFunSphere(M, X, Y)
        dist = M.dist(X,Y);
        xydiff = X - Y;
        normdiff = norm(X-Y);
        if normdiff ~= 0
            val = xydiff/(normdiff*sqrt(1-normdiff^2/4));
            val = val * 2 * dist;
            val = val - X*(X(:).'*val(:));
        end
        val = val * 0.01;
    end

    
    
        function val = costFun(data, x)
            Inner = - x.'*data;
            val = max(Inner(:));
        end

        function val = gradFun(data, x)
            Inner = - x.'*data;
            [maxval,pos] = max(Inner(:));
            val = - data(:, pos);
        end
        function displayinfo(stats)
        finalcost = stats(end).cost;
        for numcost = 1 : length([stats.cost])
            stats(numcost).cost = stats(numcost).cost - finalcost;
        end
        
        figure;
        subplot(2,2,1)
        semilogy([stats.gradnorm], '.-');
        xlabel('Iter');
        ylabel('GradNorms');
        
        titletest = sprintf('Time: %f', stats(end).time);
        title(titletest);
%         
%         subplot(2,2,2)
%         plot([stats.alpha], '.-');
%         xlabel('Iter');
%         ylabel('Alphas');
        
        subplot(2,2,3)
        semilogy([stats.stepsize], '.-');
        xlabel('Iter');
        ylabel('stepsizes');
        
        subplot(2,2,4)
        semilogy([stats.cost], '.-');
        xlabel('Iter');
        ylabel('costs');
    end

    function displaystats(stats)
        
        finalcost = stats.costs(end);
        for numcost = 1 : length(stats.costs)
            stats.costs(1,numcost) = stats.costs(1,numcost) - finalcost;
        end
        figure;
        
        subplot(2,2,1)
        semilogy(stats.gradnorms, '.-');
        xlabel('Iter');
        ylabel('GradNorms');
        
        titletest = sprintf('Time: %f', stats.time);
        title(titletest);
        
        subplot(2,2,2)
        plot(stats.alphas, '.-');
        xlabel('Iter');
        ylabel('Alphas');
        
        subplot(2,2,3)
        semilogy(stats.stepsizes, '.-');
        xlabel('Iter');
        ylabel('stepsizes');
        
        subplot(2,2,4)
        semilogy(stats.costs, '.-');
        xlabel('Iter');
        ylabel('costs');
    end
end