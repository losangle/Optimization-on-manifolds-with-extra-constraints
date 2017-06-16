dim = 3;
A = randn(dim,dim);
A = A + A.';
fprintf('A is: \n ')
disp(A)
fprintf('eigenvalue of A is: \n ')
disp(eig(A))
cost = @(x) (x'*A*x);
grad = @(x) 2*A*x;

% cost = @(x) sin(x'*x);
% grad = @(x) cos(x'*x)*2*x;

%euclidean(cost,grad,dim)


% Create the problem structure.
manifold = spherefactory(dim);
problem.M = manifold;

% Define the problem cost function and its Euclidean gradient.
problem.cost  = cost;
problem.egrad = grad;

% Solve.
[x, xcost, info, options] = trustregions(problem);
 
disp(norm(grad(x)-x*(grad(x)'*x)))