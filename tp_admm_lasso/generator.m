m = 1000;
n = 100000;
p = 100/m/n;

x0 = sprandn(n, 1, p);
A = sprandn(m, n, p);
A = A*spdiags(1./sqrt(sum(A.^2))', 0, n, n);
b = A*x0 + sqrt(0.0001)*randn(m,1);

lambda_max = norm(A'*b, 'inf');
lambda = 0.1*lambda_max;

mat_file = matfile('data.mat','Writable', true);
mat_file.A = A;
mat_file.b = b;
mat_file.x0 = x0;
mat_file.lambda_max = lambda_max;
mat_file.lambda = lambda;