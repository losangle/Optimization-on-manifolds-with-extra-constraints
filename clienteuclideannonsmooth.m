function clienteuclideannonsmooth
    dim = 2;
    w = 8;
    cost = @(x) (1-x(1))^2+w*abs(x(2)-x(1)^2);
    grad = @(x) [sign(x(2)-x(1)^2)*w*(-2*x(1)); sign(x(2)-x(1)^2)*w-2*(1-x(2))];


    manifold = euclideanfactory(dim);
    problem.M = manifold;

    % Define the problem cost function and its Euclidean gradient.
    problem.cost  = cost;
    problem.egrad = grad;
    
    x = problem.M.rand();
    
    options.linesearchVersion = 4;
    options.memory = 200;
    [gradnorms, alphas, x, time] = euclideannonsmooth(problem,x,options);
 
    disp(x)
    figure;
    semilogy(gradnorms, '.-');
    xlabel('Iteration number - Nonsmooth algo');
    ylabel('Norm of the gradient of f');

end