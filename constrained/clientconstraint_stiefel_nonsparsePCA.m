function clientconstraint_stiefel_nonsparsePCA
close all; clc; clear all;
% rng(11);
Dim = 6;
dim = 3;

N_in = 80;
N_out = 20;
Lambda_in = eye(dim);
for c = 1 : dim
    Lambda_in(c,c) = c;
end
% Lambda_in(2,2) = 0;
V = zeros(Dim, dim);
V(1:dim,1:dim) = eye(dim);
% theta = -1.5;
% V = [cos(theta) , -sin(theta); sin(theta), cos(theta)];
Sig_in = V*Lambda_in*V.';
%     Sig_out = randn(Dim);
%     Sig_out = Sig_out.'*Sig_out;
Sig_out = eye(Dim);
X= zeros(N_in+N_out, Dim);
mu = zeros(Dim, 1);
X(1: N_in, :) = mvnrnd(mu, Sig_in, N_in);
X(N_in+1: N_in+N_out, :) = mvnrnd(mu, Sig_out/rank(Sig_out), N_out);
X = X.';


manifold = stiefelfactory(Dim, dim);
problem.M = manifold;
problem.cost = @(u) costfun(u);
problem.egrad = @(u) gradfun(u);

% checkgradient(problem);

constraints_cost = cell(1, Dim*dim);
for row = 1: Dim
    for col = 1: dim
        constraints_cost{(col-1)*Dim + row} = @(U) U(row, col);
    end
end

constraints_grad = cell(1, Dim * dim);
for row = 1: Dim
    for col = 1: dim
        constraintgrad = zeros(Dim, dim);
        constraintgrad(row, col) = 1;
        constraints_grad{(col-1)*Dim + row} = @(U) constraintgrad;
    end
end

problem.ineq_constraint_cost = constraints_cost;
problem.ineq_constraint_grad = constraints_grad;


% for i = 1:Dim * dim
%     newproblem.M = manifold;
%     newproblem.cost = constraints_cost{i};
%     newproblem.egrad = constraints_grad{i};
%     checkgradient(newproblem);
% end

x0 = problem.M.rand();
% x0 = zeros(size(problem.M.rand()));
% x0(Dim-dim+1:Dim ,1:dim) = eye(dim);
options = [];

xfinal = alm(problem, x0, options);
% xfinal = exactpenalty(problem, x0, options);
% xfinal = logbarrier(problem, x0, options);

M = problem.M;
        figure
    subplot(2,3,1);
    surfprofile(problem, xfinal, M.randvec(xfinal), M.randvec(xfinal));
    subplot(2,3,2);
    surfprofile(problem, xfinal, M.randvec(xfinal), M.randvec(xfinal));
    subplot(2,3,3);
    surfprofile(problem, xfinal, M.randvec(xfinal), M.randvec(xfinal));
    subplot(2,3,4);
    surfprofile(problem, xfinal, M.randvec(xfinal), M.randvec(xfinal));
    subplot(2,3,5);
    surfprofile(problem, xfinal, M.randvec(xfinal), M.randvec(xfinal));
    subplot(2,3,6);
    surfprofile(problem, xfinal, M.randvec(xfinal), M.randvec(xfinal));

    what(X, xfinal);

% figure
% scatter(X(1, :), X(2, :));
% hold on
% plot([0;xfinal(1)], [0;xfinal(2)], 'LineWidth', 5);
% axis([-3 3 -3 3])
% hold off

    function val = costfun(u)
        val = -.5 * norm(u.' * X, 'fro')^2;
    end

    function val = gradfun(u)
        val = -X * (u.' * X).';
    end

end

function what(X, xfinal)
    check  = 1;
end
