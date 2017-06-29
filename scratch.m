clear all, close all, clc;

A=[1,2,3,4;2,3,4,5;3,4,5,6;4,5,6,7];
A(eye(size(A,1))==1) = -2;
A
[Y,I] = max(A(:))

floor(11/4)

B = [1,4,5;2,3,6];
D = zeros(size(B))
D(:,1) = B(:,2)
D(:,2) = B(:,1)