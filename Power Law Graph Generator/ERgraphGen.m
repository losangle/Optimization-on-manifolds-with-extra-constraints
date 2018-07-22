function [A, L] = ERgraphGen(n, p)
    A = zeros(n, n);
    for row = 1:n
         for col = row+1:n
            coin = unifrnd(0,1);
            if coin < p
                A(row, col) = 1;
            end
            A(col, row) = A(row, col);
         end
    end
    sumA = sum(A, 2);
    L = zeros(n, n);
    for row = 1:n
        L(row, row) = sumA(row, 1);
    end
    L = L - A;
end