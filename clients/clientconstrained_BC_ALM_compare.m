function data = clientconstraint_BC_ALM_compare(L, rankY, methodoptions, specifier)

data = NaN(3,5);
[N, ~] = size(L);
manifold = obliquefactory(rankY, N);
problem.M = manifold;
problem.cost = @(u) costFun(u);
problem.egrad = @(u) gradFun(u);
x0 = problem.M.rand();

%     DEBUG only
%     checkgradient(problem);

%-------------------------Set-up Constraints-----------------------
colones = ones(N, 1);

eq_constraints_cost = cell(1,1);
eq_constraints_cost{1} = @(U) norm(U*colones,2)^2;

eq_constraints_grad = cell(1,1);
eq_constraints_grad{1} = @(U) meanzero_eq_constraint(U);

problem.eq_constraint_cost = eq_constraints_cost;
problem.eq_constraint_grad = eq_constraints_grad;

%     Debug Only
%     checkconstraints(problem)

condet = constraintsdetail(problem);

%     ------------------------- Solving ---------------------------
    options = methodoptions;
    
    for ialm = 1: 5
        %ALM
        fprintf('Starting ALM \n');
        timetic = tic();
        options.bound = 10 * 5^(ialm-1);
        [xfinal, info] = almbddmultiplier(problem, x0, options);
        time = toc(timetic);
        
        [maxviolation, meanviolation, cost] = evaluation(problem, xfinal, condet);
        maxviolation = max(maxviolation, manifoldViolation(xfinal));
        data(1, ialm) = maxviolation;
        data(2, ialm) = cost;
        data(3, ialm) = time;
    end
    
     %------------------------sub functions-----------
    
    function stop = outfun(x, optimValues, state)
        stop = false;
        if toc(timetic) > methodoptions.maxtime
            stop = true;
        end
    end 
     
    function [f, g] = costFunfmincon(v)
        Y = reshape(v, [rankY, N]);
        f = trace((Y) * L * (Y.') );
        if nargout > 1
            g = Y*L + Y*L.';
            g = g(:);
        end
    end 

    function [c, ceq, gradc, gradceq] = nonlcon(v)
        Y = reshape(v, [rankY, N]);
        ceq = zeros(N+1,1);
        for rowCeq = 1: N
            ceq(rowCeq,1) = Y(:,rowCeq).'*Y(:,rowCeq) - 1;
        end
        ceq(N+1,1) = colones.' *(Y.')* Y * colones;
        c = [];
        if nargout > 2
            gradc = [];
            gradceq = zeros(rankY*N, N+1);
            grad = 2* repmat(Y*colones, 1, N);
            gradceq(:, N+1) = grad(:);
            for rowCeq = 1: N
                grad = zeros(rankY, N);
                grad(:, rowCeq) = 2*Y(:, rowCeq);
                gradceq(:, rowCeq) = grad(:);
            end
        end
    end

    function val = meanzero_eq_constraint(U)
        val = 2* repmat(U*colones, 1, N);
    end

    function val = costFun(u)
        val = trace((u) * L * (u.') );
    end

    function val = gradFun(u)
        val = u*L + u*L.';
    end

    function manvio = manifoldViolation(x)
        %Oblique Factory:
        manvio = max(abs(diag(x.'*x)-colones));
    end
end

