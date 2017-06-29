function clientnonsmooth
    cost = @(X) costFun(X);
    grad = @(X) gradFun(X);

    % Create the problem structure.
    manifold = obliquefactory(100,10);
    problem.M = manifold;

    % Define the problem cost function and its Euclidean gradient.
    problem.cost  = cost;
    problem.egrad = grad;

    %Set options
    options.linesearchVersion = 4;
    options.memory = 2000;

    xCur = problem.M.rand();

    bfgsnonsmooth(problem, xCur, options);
    
%     bfgsIsometric(problem, xCur, options);
%    bfgsClean(problem, xCur, options);
    %trustregions(problem, xCur, options);
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
        function val = costFun(X)
            Inner = X.'*X;
            Inner(eye(size(Inner,1))==1) = -2;
            val = max(Inner(:));
        end

        function val = gradFun(X)
            Inner = X.'*X;
            m = size(Inner,1);
            Inner(eye(m)==1) = -2;
            [maxval,pos] = max(Inner(:));
            i = mod(pos-1,m)+1;
            j = floor((pos-1)/m)+1;
            val = zeros(size(X));
            val(:,i) = X(:,j);
            val(:,j) = X(:,i);
        end
end