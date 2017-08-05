function clienteuclideannonsmooth

    dim = 2;
    w = 100;
    cost = @(x) (1-x(2))^2+w*abs(x(2)-x(1)^2);
    grad = @(x) [sign(x(2)-x(1)^2)*w*(-2*x(1)); sign(x(2)-x(1)^2)*w-2*(1-x(2))];
    
    
    manifold = euclideanfactory(dim);
    problem.M = manifold;

    % Define the problem cost function and its Euclidean gradient.
    problem.cost  = cost;
    problem.egrad = grad;
    
    x = problem.M.rand();
    
    options.memory = 200;
%     bfgsSmooth(problem, x, options);
%     trustregions(problem, x, options);
%   [gradnorms, alphas, stepsizes, costs, xCur, time] = euclideannonsmoothlm(problem,x,options);
     [gradnorms, alphas, stepsizes, costs, xCur, time] = euclideannonsmooth(problem,x,options);
%     [gradnorms, alphas, stepsizes, costs, xCur, time] = bfgsnonsmooth(problem, x, options);
     [xLate, cost, info, options] = blockbfgs(problem, x, options);

    disp(xCur)
    disp(xLate)
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
    semilogy(costs, '.-');
    xlabel('Iter');
    ylabel('costs');

end