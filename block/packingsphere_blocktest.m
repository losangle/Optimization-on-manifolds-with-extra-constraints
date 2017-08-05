function clientnonsmoothClean
clear all;
close all;
clc;
%     rng(7141981);
    d = 3;
    n = 24;
    % Create the problem structure.
    manifold = obliquefactory(d,n);
    problem.M = manifold;
    discrepency = 1e-6;
    
    cost = @(X) costFun(X);
    subgrad = @(X) subgradFun(manifold, X, discrepency);
    subgradTwoNorm = @(X, discre, handle) subgradFun(manifold, X, discre);
    subgradPnorm = @(X, discre, handle) subgradFunPnorm(manifold, X, discre, handle);
    gradFunc = @(X) gradFun(X);
    

    % Define the problem cost function and its Euclidean gradient.
    problem.cost  = cost;
    problem.grad = gradFunc;
    

%    checkgradient(problem);

    %Set options
    xCur = problem.M.rand();
    
%     options = [];
%     profile clear;
%     profile on;
%     problem.subgrad = subgradTwoNorm;
%     fprintf('___________________________Two Norm___________________\n')
%     [X, cost, stats, options] = bfgsnonsmoothwen(problem, xCur, options);
% %     [stats, X]  = bfgsnonsmoothCleanCompare(problem, xCur, options);
% %     [stats, X]  = bfgsnonsmoothClean(problem, xCur, options);
%     
% %     profile off;
% %     profile report
%     
%     figure
%     h = logspace(-15, 1, 101);
%     vals = zeros(1, 101);
%     for iter = 1:101
%         vals(1,iter) = problem.M.norm(X, subgradFun(problem.M, X, h(iter)));
%     end
%     loglog(h, vals)
%     
%     displayinfo(stats)

    
    
%     problem.subgrad= subgradPnorm;
    problem.subgrad = @(X, discre, handle) gradFun(X);
    options.partialP = @(k, memory) min(k, memory);
    fprintf('___________________________P Norm full handle___________________\n')
    
%     profile clear;
%     profile on;

    options.maxiter = 20000;
%     [X, cost, stats, options] = bfgsnonsmoothwen(problem, xCur, options);
%     [X, cost, stats, options] = blockbfgs(problem, xCur, options);
%      [X, cost, stats, options] = rlbfgs(problem, xCur, options);
%     [X, cost, stats, options] = frogbfgs(problem, xCur, options);
    [X, cost, stats, options] = rerealization(problem, xCur, options);

%     profile off;
%     profile report

    figure
    h = logspace(-15, 1, 101);
    vals = zeros(1, 101);
    for iter = 1:101
        vals(1,iter) = problem.M.norm(X, subgradFun(problem.M, X, h(iter)));
    end
    loglog(h, vals)
    
    displayinfo(stats)
%     displayinfo(info)

    M = problem.M;
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



%     problem.subgrad= subgradPnorm;
%     options =[];
%     options.partialP = @(k, memory) partialPhandle(k, memory);
%     fprintf('___________________________P Norm___________________\n')
%     
%     profile clear;
%     profile on;
% 
%     
%     [X, cost, stats, options] = bfgsnonsmoothwen(problem, xCur, options);
%     
%     profile off;
%     profile report
% 
%     figure
%     h = logspace(-15, 1, 101);
%     vals = zeros(1, 101);
%     for iter = 1:101
%         vals(1,iter) = problem.M.norm(X, subgradFun(problem.M, X, h(iter)));
%     end
%     loglog(h, vals)
%     
%     displayinfo(stats)
    
        drawsphere(X, d);
    
    
    function val = partialPhandle(k, memory)
        l = min(k, memory);
        if l <= 3
            val = l;
        else
            val = floor(2*sqrt(l));
        end
    end
    
    
    function val = costFun(X)
        Inner = X.'*X;
        Inner(1:size(Inner,1)+1:end) = -2;
        val = max(Inner(:));
    end

    function u = subgradFunPnorm(M, X, discrepency, P_operator)
        if (~exist('discrepency', 'var'))
            discrepency = 1e-5;
        end
        counter = 0;
        Inner = X.'*X;
        m = size(Inner, 1);
        Inner(1: m+1: end) = -2;
        [maxval, unusedpos] = max(Inner(:));
        pairs = zeros(m*m, 2);
        for row = 1: m
            for col = row+1:m
                if abs(Inner(row, col)-maxval) <= discrepency
                    counter = counter +1;
                    pairs(counter, :) = [row, col];
                end
            end
        end
        grads = cell(1, counter);
        for iterator = 1 : counter
            val = zeros(size(X));
            pair = pairs(iterator, :);
            Innerprod = X(:, pair(1, 1)).'*X(:, pair(1, 2));
            val(:, pair(1, 1)) = X(:, pair(1, 2)) - Innerprod*X(:,pair(1, 1));
            val(:, pair(1, 2)) = X(:, pair(1, 1)) - Innerprod*X(:,pair(1, 2));
            grads{iterator} = val;
        end
        fprintf('counter is %d', counter);
        [u_norm, coeffs, u, nonposdef] = smallestinconvexhullpnorm(M, X, grads, min(discrepency, 1e-15), P_operator);
        % u_norm == 0 iff real(sqrt(2*cost)) == 0 iff cost is negative
        % since cost is real with real gram matrix
        if nonposdef
            u = NaN;
        end
    end

    function u = subgradFun(M, X, discrepency)
        if (~exist('discrepency', 'var'))
            discrepency = 1e-5;
        end
        counter = 0;
        max_total_counter = 1000;
        pairs = [];
        Inner = X.'*X;
        m = size(Inner, 1);
        Inner(1: m+1: end) = -2;
        [maxval,pos] = max(Inner(:));
        pairs = zeros(m*m, 2);
        for row = 1: m
            for col = row+1:m
                if abs(Inner(row, col)-maxval) <= discrepency
                    counter = counter +1;
                    pairs(counter, :) = [row, col];
                end
            end
        end
        counter = min(counter, max_total_counter);
        grads = cell(1, counter);
        for iterator = 1 : counter
            val = zeros(size(X));
            pair = pairs(iterator, :);
            Innerprod = X(:, pair(1, 1)).'*X(:, pair(1, 2));
            val(:, pair(1, 1)) = X(:, pair(1, 2)) - Innerprod*X(:,pair(1, 1));
            val(:, pair(1, 2)) = X(:, pair(1, 1)) - Innerprod*X(:,pair(1, 2));
            grads{iterator} = val;
        end
        fprintf('counter is %d', counter);
        [u_norm, coeffs, u] = smallestinconvexhull(M, X, grads, min(discrepency, 1e-15));
    end

    function val = gradFun(X)
        Inner = X.'*X;
        m = size(Inner,1);
        Inner(1:m+1:end) = -2;
        [maxval,pos] = max(Inner(:));
        i = mod(pos-1,m)+1;
        j = floor((pos-1)/m)+1;
        val = zeros(size(X));
%         val(:,i) = X(:,j);
%         val(:,j) = X(:,i);
        Innerprod = X(:, i).'*X(:, j);
        val(:, i) = X(:, j) - Innerprod*X(:,i);
        val(:, j) = X(:, i) - Innerprod*X(:,j);
    end

    function drawsphere(X, dim)
        maxdot = costFun(X);
        
        if dim == 3
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
            Y = 1.02*X;
            plot3(Y(1, :), Y(2, :), Y(3, :), 'r.', 'MarkerSize', 25);
            % And connect the points which are at minimal distance,
            % within some tolerance.
            min_distance = real(acos(maxdot));
            connected = real(acos(X.'*X)) <= 1.20*min_distance;
            [Ic, Jc] = find(triu(connected, 1));
            for k = 1 : length(Ic)
                vertex1 = Ic(k); vertex2 = Jc(k);
                plot3(Y(1, [vertex1 vertex2]), Y(2, [vertex1 vertex2]), Y(3, [vertex1 vertex2]), 'k-');
            end
            hold off;
        end
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
        
        if exist('stats.alphas', 'var') 
        subplot(2,2,2)
        plot([stats.alpha], '.-');
        xlabel('Iter');
        ylabel('Alphas');
        end
        
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
        
        if exist('stats.alphas', 'var') 
                subplot(2,2,2)
                plot(stats.alphas, '.-');
                xlabel('Iter');
                ylabel('Alphas');
        end
        

        
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

