clear all, close all, clc;

M = obliquefactory(5,3);
x1 = M.rand();
x2 = M.rand();
u1 = M.randvec(x1);
u2 = M.randvec(x1);
inner = M.inner(x1, u1, u2)
v1 = M.isotransp(x1, x2, u1);
v2 = M.isotransp(x1, x2, u2);
inner2  = M.inner(x2, v1, v2)
inner2 = M.inner(x2, x2, v2)