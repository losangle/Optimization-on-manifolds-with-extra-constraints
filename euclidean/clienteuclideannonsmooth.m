function clienteuclideannonsmooth

    dim = 2;
    w = 10;
    cost = @(x) (1-x(2))^2+w*abs(x(2)-x(1)^2);
    grad = @(x) [sign(x(2)-x(1)^2)*w*(-2*x(1)); sign(x(2)-x(1)^2)*w-2*(1-x(2))];
    
    
    manifold = euclideanfactory(dim);
    problem.M = manifold;

    % Define the problem cost function and its Euclidean gradient.
    problem.cost  = cost;
    problem.egrad = grad;
    
    x = problem.M.rand();
    
    options.memory = 200;
%    [gradnorms, alphas, stepsizes, costs, xCur, time] = euclideannonsmoothlm(problem,x,options);
     [gradnorms, alphas, stepsizes, costs, xCur, time] = euclideannonsmooth(problem,x,options);

%       Smooth Version does not work
%     euclideansmooth (cost,grad,dim)

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
    semilogy(costs, '.-');
    xlabel('Iter');
    ylabel('costs');

end