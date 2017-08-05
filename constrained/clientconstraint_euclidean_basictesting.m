function clientconstraint_euclidean_basictesting
manifold = euclideanfactory(2);
problem.M = manifold;
problem.cost = @(X) costfun(X);
problem.grad = @(X) gradfun(X);
constraints_cost = cell(1, 4);
constraints_cost{1} = @(X) X(1) + 1;
constraints_cost{2} = @(X) 1 - X(1);
constraints_cost{3} = @(X) X(2) + 1;
constraints_cost{4} = @(X) 1 - X(2);

constraints_grad = cell(1, 4);
constraints_grad{1} = @(X) [1; 0];
constraints_grad{2} = @(X) [-1; 0];
constraints_grad{3} = @(X) [0; 1];
constraints_grad{4} = @(X) [0; -1];

problem.ineq_constraint_cost = constraints_cost;
problem.ineq_constraint_grad = constraints_grad;
% for i = 1:4
%     newproblem.M = manifold;
%     newproblem.cost = constraints_cost{i};
%     newproblem.grad = constraints_grad{i};
% end

x0 = problem.M.rand();
options = [];
alm(problem, x0, options);

    function val = costfun(x)
        val = -norm(x, 2)^2/2;
    end

    function val = gradfun(x)
        val = -x;
    end

end


