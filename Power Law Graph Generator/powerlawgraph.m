function L = powerlawgraph(seedsize, probER, Dim, mlink)
% Returns the Laplacian of the graph. It is a sparse matrix.
% seedsize is the size of the seed to get the powerlaw graph
% probER is the probability to connect two nodes in ER graph
% Dim is the number of nodes of the powerlaw graph
% mlink is the number of links a new node is allowed to connect
% in B-A algorithm

    % Generate seed graph until every node has at least one edge
    while (1)
        [A, ~] = ERgraphGen( 2* max(seedsize, mlink), probER);
        deg = sum(A,2);
        mindeg = min(deg);
        if mindeg > 0
            break;
        end
    end
    
    Adj = SFNG(Dim, mlink, A);
    sumAdj = sum(Adj, 2);
    L = -Adj;
    [n, ~] = size(L);
    for row = 1 : n
        L(row, row) = sumAdj(row, 1);
    end
    L = sparse(double(L));
end