manifold = spherefactory(10);
problem.M = manifold;
M = problem.M;
x = M.rand();
v = M.randvec(x);
M.norm(x,v)
y = M.rand();
u = M.transp(x,y,v);
M.norm(x,u)