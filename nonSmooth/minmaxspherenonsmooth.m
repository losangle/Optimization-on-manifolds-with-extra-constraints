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
    
    %Set options
    options.linesearchVersion = 4;
    options.memory = 400;

    X1 = problem.M.rand();
    X2 = problem.M.rand();
    X3 = problem.M.rand();
    options.assumedoptX = problem.M.rand();
    while (1)
        xCur = problem.M.rand();
        [gradnorms, alphas, stepsizes, costs, distToAssumedOptX, X1, time] = bfgsnonsmooth(problem, xCur, options);
        if M.dist(X1,X2)+M.dist(X1,X3)+M.dist(X2,X3) <= 1e-5
            break;
        else
            X3 = X2;
            X2 = X1;
        end
    end
    options.assumedoptX = X1;
    xCur = problem.M.rand();
    [gradnorms, alphas, stepsizes, costs, distToAssumedOptX, xCur, time] = bfgsnonsmooth(problem, xCur, options);
    disp(xCur)
    figure;
    
    subplot(2,2,1)
    semilogy(gradnorms, '.-');
    xlabel('Iter');
    ylabel('GradNorms');

    titletest = sprintf('Time: %f', time);
    title(titletest);
    
    subplot(2,2,2)
    plot(alphas, '.-');
    xlabel('Iter');
    ylabel('Alphas');

    subplot(2,2,3)
    semilogy(stepsizes, '.-');
    xlabel('Iter');
    ylabel('stepsizes');

    subplot(2,2,4)
%     plot(costs, '.-');
    semilogy(distToAssumedOptX, '.-');
    xlabel('Iter');
    ylabel('costs');

%     bfgsIsometric(problem, xCur, options);
%    bfgsClean(problem, xCur, options);
%     %trustregions(problem, xCur, options);
%     options.maxiter = 20000;
%     steepestdescent(problem, xCur, options);
    
%     profile clear;
%     profile on;
% 
%     bfgsClean(problem,xCur,options);
% 
% 
%     profile off;
%     profile report

    % This can change, but should be indifferent for various
    % solvers.
    % Integrating costGrad and cost probably halves the time
        function val = costFun(data, x)
            Inner = - x.'*data;
            val = max(Inner(:));
        end

        function val = gradFun(data, x)
            Inner = - x.'*data;
            [maxval,pos] = max(Inner(:));
            val = - data(:, pos);
        end
end
