    function [maxviolation, meanviolation, cost] = evaluation(problem, x, condet)
        maxviolation = 0;
        meanviolation = 0;
        cost = getCost(problem, x);

        for numineq = 1: condet.n_ineq_constraint_cost
            costhandle = problem.ineq_constraint_cost{numineq};
            cost_at_x = costhandle(x);
            maxviolation = max(maxviolation, cost_at_x);
            meanviolation = meanviolation + max(0, cost_at_x);
        end
        
        for numeq = 1: condet.n_eq_constraint_cost
            costhandle = problem.eq_constraint_cost{numeq};
            cost_at_x = abs(costhandle(x));
            maxviolation = max(maxviolation, cost_at_x);
            meanviolation = meanviolation + cost_at_x;
        end

        meanviolation = meanviolation / (condet.n_ineq_constraint_cost + condet.n_eq_constraint_cost);
    end