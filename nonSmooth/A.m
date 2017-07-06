function A(problem, xFinal)
    manifold = problem.M;

    B(problem, xFinal);
end

function u = subgradFun(M, X, discrepency)
    counter = 0;
    pairs = [];
    Inner = X.'*X;
    m = size(Inner, 1);
    Inner(1: m+1: end) = -2;
    [maxval,pos] = max(Inner(:));
    pairs = zeros(m*m, 2);
    for row = 1: m
        for col = row+1:m
            if abs(Inner(row, col)-maxval) <= discrepency
                counter = counter +1;
                pairs(counter, :) = [row, col];
            end
        end
    end
    pairs = pairs(1:counter, :)
    grads = cell(1, counter);
    for iterator = 1 : counter
        val = zeros(size(X));
        pair = pairs(iterator, :);
        Innerprod = X(:, pair(1, 1)).'*X(:, pair(1, 2));
        val(:, pair(1, 1)) = X(:, pair(1, 2)) - Innerprod*X(:,pair(1, 1));
        val(:, pair(1, 2)) = X(:, pair(1, 1)) - Innerprod*X(:,pair(1, 2));
        grads{iterator} = val;
    end
    [u_norm, coeffs, u] = smallestinconvexhull(M, X, grads);
end

function B(problem, x)
        what = 1;
end