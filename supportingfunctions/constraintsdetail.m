function condet = constraintsdetail(problem)
% It takes in an problem and returns a struct with the
% following fields:
% 
% Booleans:
% has_ineq_cost: True if problem has inequality cost functions. (e.g)
% has_ineq_grad
% has_eq_cost
% has_eq_grad
% 
% Int:
% n_ineq_constraint_cost: Number of cost function in inequality constraints.
% n_ineq_constraint_grad
% n_eq_constraint_cost
% n_eq_constraint_grad
% 
% It displays warning if the number of cost and grad functiosn do not match.
%
%
% This file is part of Manopt: www.manopt.org.
% Original author: Changshuo Liu, September 3, 2017.


    condet.has_ineq_cost = isfield(problem, 'ineq_constraint_cost');
    condet.has_ineq_grad = isfield(problem, 'ineq_constraint_grad');
    condet.has_eq_cost = isfield(problem, 'eq_constraint_cost');
    condet.has_eq_grad = isfield(problem, 'eq_constraint_grad');
    
    if condet.has_ineq_cost
        condet.n_ineq_constraint_cost  = length(problem.ineq_constraint_cost);
    else
        condet.n_ineq_constraint_cost = 0;
    end
    if condet.has_ineq_grad
        condet.n_ineq_constraint_grad  = length(problem.ineq_constraint_grad);
    else
        condet.n_ineq_constraint_grad  = 0;
    end
    
    if condet.has_eq_cost
        condet.n_eq_constraint_cost  = length(problem.eq_constraint_cost);
    else
        condet.n_eq_constraint_cost = 0;
    end
    if condet.has_eq_grad
        condet.n_eq_constraint_grad  = length(problem.eq_constraint_grad);
    else 
        condet.n_eq_constraint_grad = 0;
    end

    if (condet.n_ineq_constraint_cost ~= condet.n_ineq_constraint_grad)
        warning('checkconstraints:number',['the number of cost functions of'...
            'inequality constraints do not match the number of gradient functions']);
    end
    
    if (condet.n_eq_constraint_cost ~= condet.n_eq_constraint_grad)
        warning('checkconstraints:number',['the number of cost functions of'...
            'equality constraints do not match the number of gradient functions']);
    end
    
    
end