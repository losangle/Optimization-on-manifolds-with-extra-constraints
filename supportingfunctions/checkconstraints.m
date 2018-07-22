function checkconstraints(problem, x0, d)

    has_d = exist('d', 'var') && ~isempty(d);
    has_x = exist('x0', 'var')&& ~isempty(x0);
    has_ineq_cost = isfield(problem, 'ineq_constraint_cost');
    has_ineq_grad = isfield(problem, 'ineq_constraint_grad');
    has_eq_cost = isfield(problem, 'eq_constraint_cost');
    has_eq_grad = isfield(problem, 'eq_constraint_grad');
    
    if has_ineq_cost
        n_ineq_constraint_cost  = length(problem.ineq_constraint_cost);
    else
        n_ineq_constraint_cost = 0;
    end
    if has_ineq_grad
        n_ineq_constraint_grad  = length(problem.ineq_constraint_grad);
    else
        n_ineq_constraint_grad  = 0;
    end
    
    if has_eq_cost
        n_eq_constraint_cost  = length(problem.eq_constraint_cost);
    else
        n_eq_constraint_cost = 0;
    end
    if has_eq_grad
        n_eq_constraint_grad  = length(problem.eq_constraint_grad);
    else 
        n_eq_constraint_grad = 0;
    end

    if (n_ineq_constraint_cost ~= n_ineq_constraint_grad)
        warning('checkconstraints:number',['the number of cost functions of'...
            'inequality constraints do not match the number of gradient functions']);
    end
    
    if (n_eq_constraint_cost ~= n_eq_constraint_grad)
        warning('checkconstraints:number',['the number of cost functions of'...
            'equality constraints do not match the number of gradient functions']);
    end
    
    for iter = 1:n_ineq_constraint_cost
        newproblem.M = problem.M;
        newproblem.cost = problem.ineq_constraint_cost{iter};
        newproblem.egrad = problem.ineq_constraint_grad{iter};
        if has_x
            if has_d
                checkgradient(newproblem, x0, d);
            else
                checkgradient(newproblem, x0);
            end
        else
            checkgradient(newproblem);
        end
    end
    
    for iter = 1:n_eq_constraint_cost
        newproblem.M = problem.M;
        newproblem.cost = problem.eq_constraint_cost{iter};
        newproblem.egrad = problem.eq_constraint_grad{iter};
        if has_x
            if has_d
                checkgradient(newproblem, x0, d);
            else
                checkgradient(newproblem, x0);
            end
        else
            checkgradient(newproblem);
        end
    end  
    
end