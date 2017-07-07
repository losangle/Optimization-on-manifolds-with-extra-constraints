function minmaxspherenonsmooth

    d = 10;
    n = 20;
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
    
    
    % Define the problem cost function and its Euclidean gradient.
    problem.cost  = cost;
    problem.egrad = grad;
    problem.reallygrad = grad;
    
    %Set options
    options = [];

    X1 = problem.M.rand();
    X2 = problem.M.rand();
    X3 = problem.M.rand();
    options.assumedoptX = problem.M.rand();
    while (1)
        xCur = problem.M.rand();
       [stats, X1] = bfgsnonsmoothClean(problem, xCur, options);
        if M.dist(X1,X2)+M.dist(X1,X3)+M.dist(X2,X3) <= 1e-3
            break;
        else
            X3 = X2;
            X2 = X1;
        end
    end
    profile clear;
    profile on;
    options.assumedoptX = X1;
    xCur = problem.M.rand();
    xCur(1,1) = abs(xCur(1,1));
   [stats, XCur] = bfgsnonsmoothClean(problem, xCur, options);
    
    profile off;
    profile report
    
    disp(xCur)
    displaystats(stats)

    
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
    
    
    
        function val = costFun(data, x)
            Inner = - x.'*data;
            val = max(Inner(:));
        end

        function val = gradFun(data, x)
            Inner = - x.'*data;
            [maxval,pos] = max(Inner(:));
            val = - data(:, pos);
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
