clear all, close all, clc;

M = spherefactory(5);
x1 = M.rand();
x2 = M.rand();
u1 = M.randvec(x1);
u2 = M.randvec(x1);
inner1 = M.inner(x1,u1,u2);
fprintf('%.16e\n',inner1)
v1 = M.isotransp(x1,x2,u1);
v2 = M.isotransp(x1,x2,u2);
fprintf('%.16e',M.inner(x2,v1,v2))

% x1 = M.rand();
% p = M.randvec(x1);
% alpha = 100000000000;
% M.norm(x1,p)*alpha
% u1 = M.randvec(x1);
% u2 = M.randvec(x1);
% inner1 = M.inner(x1,u1,u2)
% v1 = M.isotranspdir(x1,alpha,p,u1);
% v2 = M.isotranspdir(x1,alpha,p,u2);
% inner2 = M.inner(M.exp(x1,alpha,p),v1,v2)