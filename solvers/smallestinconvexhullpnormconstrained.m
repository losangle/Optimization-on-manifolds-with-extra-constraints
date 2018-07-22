function [u_norm, coeffs, u, nonposdef] = smallestinconvexhullpnormconstrained(M, x, mustPutVec, ineqVecs, eqVecs, P_operator, tol)


    % Compute the Gram matrix of the given tangent vectors
    N_ineqVecs = length(ineqVecs);
    N_eqVecs = length(eqVecs);
    N = 1 + N_ineqVecs + N_eqVecs;
    vectors = cell(1, N);
    for ind = 1:N_ineqVecs
        vectors{ind} = ineqVecs{ind};
    end
    for ind = 1:N_eqVecs
        vectors{ind + N_ineqVecs} = eqVecs{ind};
    end
    vectors{N} = mustPutVec;
    gram_Matrix = grammatrixPnorm(M, x, vectors, P_operator);

    % Solve the quadratic program.
    % If the optimization toolbox is not available, consider replacing with
    % CVX.

    if ~exist('tol', 'var') || isempty(tol)
        tol = 1e-8;
    end

    LB = zeros(N,1);
    LB(N_ineqVecs+1 : N_ineqVecs + N_eqVecs) = -1;
    UB = ones(N,1);
    A = zeros(1,N);
    A(1,N) = 1;
    
    
    
    opts = optimset('Display', 'off', 'TolFun', tol);
    [s_opt, cost_opt] ...
        = quadprog(gram_Matrix, zeros(N, 1),     ...  % objective (squared norm)
        [], [],             ...  % inequalities (none)
        A, 1,      ...  % equality (sum to 1)
        LB,        ...  % lower bounds (s_i >= 0)
        UB,         ...  % upper bounds (s_i <= 1)
        [],                 ...  % we do not specify an initial guess
        opts);

    % Norm of the smallest tangent vector in the convex hull:
    u_norm = real(sqrt(2*cost_opt));

    % Keep track of optimal coefficients
    coeffs = s_opt;

    % If required, construct the vector explicitly.
    u = lincomb(M, x, vectors, coeffs);
    
    % If grammatrix is not posdef, then cost will be negative;
    nonposdef = cost_opt < -1e-10;

    function gram_Matrix = grammatrixPnorm(M, x, vectors, P_operator)
        N_elt = numel(vectors);
        gram_Matrix = zeros(N_elt);
        
        for row = 1 : N_elt
            
            v_row = vectors{row};
            Pv_row = P_operator(v_row);
            
            gram_Matrix(row, row) = M.inner(x, v_row, Pv_row);
            
            for j = (row+1) : N_elt
                
                vj = vectors{j};
                gram_Matrix(row, j) = M.inner(x, Pv_row, vj);
                
                % For BFGS, we only work in the real case, so
                % conjugate is not considered.
                gram_Matrix(j, row) = gram_Matrix(row, j);
                
            end
            
        end
    end


end


