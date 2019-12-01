function [Xhat,T,V,H,Z,cost] = multichannelNMF(X,N,K,maxIt,drawConv,T,V,H,Z)
%
% multichannelNMF: Blind source separation based on multichannel NMF
%
% Coded by D. Kitamura (d-kitamura@ieee.org)
%
% # Original paper
% H. Sawada, H. Kameoka, S. Araki, and N. Ueda,	"Multichannel extensions
% of non-negative matrix factorization with complex-valued data," IEEE
% Transactions on Audio, Speech, and Language Processing, vol. 21, no. 5,
% pp. 971-982, May 2013.
%
% see also
% http://d-kitamura.net
%
% [syntax]
%   [T,V,H,Z,cost] = multichannelNMF(XX,N)
%   [T,V,H,Z,cost] = multichannelNMF(XX,N,K)
%   [T,V,H,Z,cost] = multichannelNMF(XX,N,K,it)
%   [T,V,H,Z,cost] = multichannelNMF(XX,N,K,it,drawConv)
%   [T,V,H,Z,cost] = multichannelNMF(XX,N,K,it,drawConv,T)
%   [T,V,H,Z,cost] = multichannelNMF(XX,N,K,it,drawConv,T,V)
%   [T,V,H,Z,cost] = multichannelNMF(XX,N,K,it,drawConv,T,V,H)
%   [T,V,H,Z,cost] = multichannelNMF(XX,N,K,it,drawConv,T,V,H,Z)
%
% [inputs]
%        XX: input 4th-order tensor (time-frequency-wise covariance matrices) (I x J x M x M)
%         N: number of sources
%         K: number of bases (default: ceil(J/10))
%        it: number of iteration (default: 300)
%  drawConv: plot cost function values in each iteration or not (true or false)
%         T: initial basis matrix (I x K)
%         V: initial activation matrix (K x J)
%         H: initial spatial covariance tensor (I x N x M x M)
%         Z: initial partitioning matrix (K x N)
%
% [outputs]
%      Xhat: output 4th-order tensor reconstructed by T, V, H, and Z (I x J x M x M)
%         T: basis matrix (I x K)
%         V: activation matrix (K x J)
%         H: spatial covariance tensor (I x N x M x M)
%         Z: partitioning matrix (K x N)
%      cost: convergence behavior of cost function in multichannel NMF (it+1 x 1)
%

% Check errors and set default values
[I,J,M,M] = size(X);
if size(X,3) ~= size(X,4)
    error('The size of input tensor is wrong.\n');
end
if (nargin < 3)
    K = ceil(J/10);
end
if (nargin < 4)
    maxIt = 300;
end
if (nargin < 5)
    drawConv = false;
end
if (nargin < 6)
    T = max(rand(I,K),eps);
end
if (nargin < 7)
    V = max(rand(K,J),eps);
end
if (nargin < 8)
    H = repmat(sqrt(eye(M)/M),[1,1,I,N]);
    H = permute(H,[3,4,1,2]); % I x N x M x M
end
if (nargin < 9)
    varZ = 0.01;
    Z = varZ*rand(K,N) + 1/N;
    Z = max( Z./sum(Z,2), eps ); % to ensure "sum_n Z_kn = 1" (using implicit expansion)
%     Z = bsxfun(@rdivide, Z, sum(Z,2)); % for prior R2016b
end
if sum(size(T) ~= [I,K]) || sum(size(V) ~= [K,J]) || sum(size(H) ~= [I,N,M,M]) || sum(size(Z) ~= [K,N])
    error('The size of input initial variable is incorrect.\n');
end
Xhat = local_Xhat( T, V, H, Z, I, J, M ); % initial model tensor

% Iterative update
fprintf('Iteration:    ');
if ( drawConv == true )
    cost = zeros( maxIt+1, 1 );
    cost(1) = local_cost( X, Xhat, I, J, M ); % initial cost value
    for it = 1:maxIt
        fprintf('\b\b\b\b%4d', it);
        [ Xhat, T, V, H, Z ] = local_iterativeUpdate( X, Xhat, T, V, H, Z, I, J, K, N, M );
        cost(it+1) = local_cost( X, Xhat, I, J, M );
    end
    figure;
    semilogy( (0:maxIt), cost );
    set(gca,'FontName','Times','FontSize',16);
    xlabel('Number of iterations','FontName','Arial','FontSize',16);
    ylabel('Value of cost function','FontName','Arial','FontSize',16);
else
    cost = 0;
    for it = 1:maxIt
        fprintf('\b\b\b\b%4d', it);
        [ Xhat, T, V, H, Z ] = local_iterativeUpdate( X, Xhat, T, V, H, Z, I, J, K, N, M );
    end
end
fprintf(' Multichannel NMF done.\n');
end

%%% Cost function %%%
function [ cost ] = local_cost( X, Xhat, I, J, M )
invXhat = local_inverse( Xhat, I, J, M );
XinvXhat = local_multiplication( X, invXhat, I, J, M );
trXinvXhat = local_trace( XinvXhat, I, J, M );
detXinvXhat = local_det( XinvXhat, I, J, M );
cost = real(trXinvXhat) - log(real(detXinvXhat)) - M;
cost = sum(cost(:));
end

%%% Iterative update %%%
function [ Xhat, T, V, H, Z ] = local_iterativeUpdate( X, Xhat, T, V, H, Z, I, J, K, N, M )
%%%%% Update T %%%%%
invXhat = local_inverse( Xhat, I, J, M );
invXhatXinvXhat = local_multiplicationXYX( invXhat, X, I, J, M );
Tnume = local_Tfrac( invXhatXinvXhat, V, Z, H, I, K, M ); % I x K
Tdeno = local_Tfrac( invXhat, V, Z, H, I, K, M ); % I x K
T = T.*max(sqrt(Tnume./Tdeno),eps);
Xhat = local_Xhat( T, V, H, Z, I, J, M );

%%%%% Update V %%%%%
invXhat = local_inverse( Xhat, I, J, M );
invXhatXinvXhat = local_multiplicationXYX( invXhat, X, I, J, M );
Vnume = local_Vfrac( invXhatXinvXhat, T, Z, H, J, K, M ); % K x J
Vdeno = local_Vfrac( invXhat, T, Z, H, J, K, M ); % K x J
V = V.*max(sqrt(Vnume./Vdeno),eps);
Xhat = local_Xhat( T, V, H, Z, I, J, M );

%%%%% Update Z %%%%%
invXhat = local_inverse( Xhat, I, J, M );
invXhatXinvXhat = local_multiplicationXYX( invXhat, X, I, J, M );
Znume = local_Zfrac( invXhatXinvXhat, T, V, H, K, N, M ); % K x N
Zdeno = local_Zfrac( invXhat, T, V, H, K, N, M ); % K x N
Z = Z.*sqrt(Znume./Zdeno);
Z = max( Z./sum(Z,2), eps ); % to ensure "sum_n Z_kn = 1" (using implicit expansion)
% Z = bsxfun(@rdivide, Z, sum(Z,2)); % for prior R2016b
Xhat = local_Xhat( T, V, H, Z, I, J, M );

%%%%% Update H %%%%%
invXhat = local_inverse( Xhat, I, J, M );
invXhatXinvXhat = local_multiplicationXYX( invXhat, X, I, J, M );
H = local_RiccatiSolver( invXhatXinvXhat, invXhat, T, V, H, Z, I, J, N, M );
Xhat = local_Xhat( T, V, H, Z, I, J, M );
end

%%% Xhat %%%
function [ Xhat ] = local_Xhat( T, V, H, Z, I, J, M )
Xhat = zeros(I,J,M,M);
for mm = 1:M*M
    Hmm = H(:,:,mm); % I x N
    Xhat(:,:,mm) = ((Hmm*Z').*T)*V;
end
end

%%% Tfrac %%%
function [ Tfrac ] = local_Tfrac( X, V, Z, H, I, K, M )
Tfrac = zeros(I,K);
for mm = 1:M*M
    Tfrac = Tfrac + real( (X(:,:,mm)*V').*(conj(H(:,:,mm))*Z') ); % using tr(AB') = sum_{i,j} (A_{ij}*B_{ij})
end
end

%%% Vfrac %%%
function [ Vfrac ] = local_Vfrac( X, T, Z, H, J, K, M )
Vfrac = zeros(K,J);
for mm = 1:M*M
    Vfrac = Vfrac + real( ((H(:,:,mm)*Z').*T)'*X(:,:,mm) ); % using tr(AB') = sum_{i,j} (A_{ij}*B_{ij})
end
end

%%% Zfrac %%%
function [ Zfrac ] = local_Zfrac( X, T, V, H, K, N, M)
Zfrac = zeros(K,N);
for mm = 1:M*M
    Zfrac = Zfrac + real( ((X(:,:,mm)*V.').*T)'*H(:,:,mm) ); % using tr(AB') = sum_{i,j} (A_{ij}*B_{ij})
end
end

%%% Riccati solver %%%
function [ H ] = local_RiccatiSolver(X, Y, T, V, H, Z, I, J, N, M)
X = reshape(permute(X, [3 4 2 1]), [M*M, J, I]); % invXhatXinvXhat, MM x J x I
Y = reshape(permute(Y, [3 4 2 1]), [M*M, J, I]); % invXhat, MM x J x I
deltaEyeM = eye(M)*(10^(-12)); % to avoid numerical instability
for n = 1:N % Riccati equation solver described in the original paper
    for i = 1:I
        ZTV = (T(i,:).*Z(:,n)')*V;
        A = reshape(Y(:,:,i)*ZTV', [M, M]);
        B = reshape(X(:,:,i)*ZTV', [M, M]);
        Hin = reshape(H(i,n,:,:), [M, M]);
        C = Hin*B*Hin;
        AC = [zeros(M), -1*A; -1*C, zeros(M)];
        [eigVec, eigVal] = eig(AC);
        ind = find(diag(eigVal)<0);
        F = eigVec(1:M,ind);
        G = eigVec(M+1:end,ind);
        Hin = G/F;
        Hin = (Hin+Hin')/2 + deltaEyeM;
        H(i,n,:,:) = Hin/trace(Hin);
    end
end
% for n = 1:N % Another solution of Riccati equation, which is slower than the above one. The calculation result coincides with that of the above calculation.
%     for i = 1:I
%         ZTV = (T(i,:).*Z(:,n).')*V; % 1 x J
%         A = reshape( Y(:,:,i)*ZTV.', [M, M] ); % M x M
%         B = sqrtm(reshape( X(:,:,i)*ZTV.', [M, M] )); % M x M
%         Hin = reshape( H(i,n,:,:), [M, M] );
%         Hin = Hin*B/sqrtm((B*Hin*A*Hin*B))*B*Hin; % solution of Riccati equation
%         Hin = (Hin+Hin')/2; % "+eye(M)*delta" should be added here for avoiding rank deficient in such a case
%         H(i,n,:,:) = Hin/trace(Hin);
%     end
% end
end

%%% MultiplicationXXX %%%
function [ XYX ] = local_multiplicationXYX( X, Y, I, J, M )
if M == 2
    XYX = zeros( I, J, M, M );
    x2 = real(X(:,:,1,2).*conj(X(:,:,1,2)));
    xy = X(:,:,1,2).*conj(Y(:,:,1,2));
    ac = X(:,:,1,1).*Y(:,:,1,1);
    bd = X(:,:,2,2).*Y(:,:,2,2);
    XYX(:,:,1,1) = Y(:,:,2,2).*x2 + X(:,:,1,1).*(2*real(xy)+ac);
    XYX(:,:,2,2) = Y(:,:,1,1).*x2 + X(:,:,2,2).*(2*real(xy)+bd);
    XYX(:,:,1,2) = X(:,:,1,2).*(ac+bd+xy) + X(:,:,1,1).*X(:,:,2,2).*Y(:,:,1,2);
    XYX(:,:,2,1) = conj(XYX(:,:,1,2));
else % slow
    XY = local_multiplication( X, Y, I, J, M );
    XYX = local_multiplication( XY, X, I, J, M );
end
end

%%% Multiplication %%%
function [ XY ] = local_multiplication( X, Y, I, J, M )
if M == 2
    XY = zeros( I, J, M, M );
    XY(:,:,1,1) = X(:,:,1,1).*Y(:,:,1,1) + X(:,:,1,2).*Y(:,:,2,1);
    XY(:,:,1,2) = X(:,:,1,1).*Y(:,:,1,2) + X(:,:,1,2).*Y(:,:,2,2);
    XY(:,:,2,1) = X(:,:,2,1).*Y(:,:,1,1) + X(:,:,2,2).*Y(:,:,2,1);
    XY(:,:,2,2) = X(:,:,2,1).*Y(:,:,1,2) + X(:,:,2,2).*Y(:,:,2,2);
elseif M == 3
    XY = zeros( I, J, M, M );
    XY(:,:,1,1) = X(:,:,1,1).*Y(:,:,1,1) + X(:,:,1,2).*Y(:,:,2,1) + X(:,:,1,3).*Y(:,:,3,1);
    XY(:,:,1,2) = X(:,:,1,1).*Y(:,:,1,2) + X(:,:,1,2).*Y(:,:,2,2) + X(:,:,1,3).*Y(:,:,3,2);
    XY(:,:,1,3) = X(:,:,1,1).*Y(:,:,1,3) + X(:,:,1,2).*Y(:,:,2,3) + X(:,:,1,3).*Y(:,:,3,3);
    XY(:,:,2,1) = X(:,:,2,1).*Y(:,:,1,1) + X(:,:,2,2).*Y(:,:,2,1) + X(:,:,2,3).*Y(:,:,3,1);
    XY(:,:,2,2) = X(:,:,2,1).*Y(:,:,1,2) + X(:,:,2,2).*Y(:,:,2,2) + X(:,:,2,3).*Y(:,:,3,2);
    XY(:,:,2,3) = X(:,:,2,1).*Y(:,:,1,3) + X(:,:,2,2).*Y(:,:,2,3) + X(:,:,2,3).*Y(:,:,3,3);
    XY(:,:,3,1) = X(:,:,3,1).*Y(:,:,1,1) + X(:,:,3,2).*Y(:,:,2,1) + X(:,:,3,3).*Y(:,:,3,1);
    XY(:,:,3,2) = X(:,:,3,1).*Y(:,:,1,2) + X(:,:,3,2).*Y(:,:,2,2) + X(:,:,3,3).*Y(:,:,3,2);
    XY(:,:,3,3) = X(:,:,3,1).*Y(:,:,1,3) + X(:,:,3,2).*Y(:,:,2,3) + X(:,:,3,3).*Y(:,:,3,3);
elseif M == 4
    XY = zeros( I, J, M, M );
    XY(:,:,1,1) = X(:,:,1,1).*Y(:,:,1,1) + X(:,:,1,2).*Y(:,:,2,1) + X(:,:,1,3).*Y(:,:,3,1) + X(:,:,1,4).*Y(:,:,4,1);
    XY(:,:,1,2) = X(:,:,1,1).*Y(:,:,1,2) + X(:,:,1,2).*Y(:,:,2,2) + X(:,:,1,3).*Y(:,:,3,2) + X(:,:,1,4).*Y(:,:,4,2);
    XY(:,:,1,3) = X(:,:,1,1).*Y(:,:,1,3) + X(:,:,1,2).*Y(:,:,2,3) + X(:,:,1,3).*Y(:,:,3,3) + X(:,:,1,4).*Y(:,:,4,3);
    XY(:,:,1,4) = X(:,:,1,1).*Y(:,:,1,4) + X(:,:,1,2).*Y(:,:,2,4) + X(:,:,1,3).*Y(:,:,3,4) + X(:,:,1,4).*Y(:,:,4,4);
    XY(:,:,2,1) = X(:,:,2,1).*Y(:,:,1,1) + X(:,:,2,2).*Y(:,:,2,1) + X(:,:,2,3).*Y(:,:,3,1) + X(:,:,2,4).*Y(:,:,4,1);
    XY(:,:,2,2) = X(:,:,2,1).*Y(:,:,1,2) + X(:,:,2,2).*Y(:,:,2,2) + X(:,:,2,3).*Y(:,:,3,2) + X(:,:,2,4).*Y(:,:,4,2);
    XY(:,:,2,3) = X(:,:,2,1).*Y(:,:,1,3) + X(:,:,2,2).*Y(:,:,2,3) + X(:,:,2,3).*Y(:,:,3,3) + X(:,:,2,4).*Y(:,:,4,3);
    XY(:,:,2,4) = X(:,:,2,1).*Y(:,:,1,4) + X(:,:,2,2).*Y(:,:,2,4) + X(:,:,2,3).*Y(:,:,3,4) + X(:,:,2,4).*Y(:,:,4,4);
    XY(:,:,3,1) = X(:,:,3,1).*Y(:,:,1,1) + X(:,:,3,2).*Y(:,:,2,1) + X(:,:,3,3).*Y(:,:,3,1) + X(:,:,3,4).*Y(:,:,4,1);
    XY(:,:,3,2) = X(:,:,3,1).*Y(:,:,1,2) + X(:,:,3,2).*Y(:,:,2,2) + X(:,:,3,3).*Y(:,:,3,2) + X(:,:,3,4).*Y(:,:,4,2);
    XY(:,:,3,3) = X(:,:,3,1).*Y(:,:,1,3) + X(:,:,3,2).*Y(:,:,2,3) + X(:,:,3,3).*Y(:,:,3,3) + X(:,:,3,4).*Y(:,:,4,3);
    XY(:,:,3,4) = X(:,:,3,1).*Y(:,:,1,4) + X(:,:,3,2).*Y(:,:,2,4) + X(:,:,3,3).*Y(:,:,3,4) + X(:,:,3,4).*Y(:,:,4,4);
    XY(:,:,4,1) = X(:,:,4,1).*Y(:,:,1,1) + X(:,:,4,2).*Y(:,:,2,1) + X(:,:,4,3).*Y(:,:,3,1) + X(:,:,4,4).*Y(:,:,4,1);
    XY(:,:,4,2) = X(:,:,4,1).*Y(:,:,1,2) + X(:,:,4,2).*Y(:,:,2,2) + X(:,:,4,3).*Y(:,:,3,2) + X(:,:,4,4).*Y(:,:,4,2);
    XY(:,:,4,3) = X(:,:,4,1).*Y(:,:,1,3) + X(:,:,4,2).*Y(:,:,2,3) + X(:,:,4,3).*Y(:,:,3,3) + X(:,:,4,4).*Y(:,:,4,3);
    XY(:,:,4,4) = X(:,:,4,1).*Y(:,:,1,4) + X(:,:,4,2).*Y(:,:,2,4) + X(:,:,4,3).*Y(:,:,3,4) + X(:,:,4,4).*Y(:,:,4,4);
else % slow
    X = reshape(permute(X, [3,4,1,2]), [M,M,I*J]); % M x M x IJ
    Y = reshape(permute(Y, [3,4,1,2]), [M,M,I*J]); % M x M x IJ
    XY = zeros( M, M, I*J );
    parfor ij = 1:I*J
        XY(:,:,ij) = X(:,:,ij)*Y(:,:,ij);
    end
    XY = permute(reshape(XY, [M,M,I,J]), [3,4,1,2]); % I x J x M x M
end
end

%%% Inverse %%%
function [ invX ] = local_inverse( X, I, J, M )
if M == 2
    invX = zeros(I,J,M,M);
    detX = X(:,:,1,1).*X(:,:,2,2) - X(:,:,1,2).*X(:,:,2,1);
    invX(:,:,1,1) = X(:,:,2,2);
    invX(:,:,1,2) = -1*X(:,:,1,2);
    invX(:,:,2,1) = conj(invX(:,:,1,2));
    invX(:,:,2,2) = X(:,:,1,1);
    invX = invX./detX; % using implicit expanion
%    invX = bsxfun(@rdivide, invX, detX); % for prior R2016b
elseif M == 3
    invX = zeros(I,J,M,M);
    detX = X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,3) + X(:,:,2,1).*X(:,:,3,2).*X(:,:,1,3) + X(:,:,3,1).*X(:,:,1,2).*X(:,:,2,3) - X(:,:,1,1).*X(:,:,3,2).*X(:,:,2,3) - X(:,:,3,1).*X(:,:,2,2).*X(:,:,1,3) - X(:,:,2,1).*X(:,:,1,2).*X(:,:,3,3);
    invX(:,:,1,1) = X(:,:,2,2).*X(:,:,3,3) - X(:,:,2,3).*X(:,:,3,2);
    invX(:,:,1,2) = X(:,:,1,3).*X(:,:,3,2) - X(:,:,1,2).*X(:,:,3,3);
    invX(:,:,1,3) = X(:,:,1,2).*X(:,:,2,3) - X(:,:,1,3).*X(:,:,2,2);
    invX(:,:,2,1) = X(:,:,2,3).*X(:,:,3,1) - X(:,:,2,1).*X(:,:,3,3);
    invX(:,:,2,2) = X(:,:,1,1).*X(:,:,3,3) - X(:,:,1,3).*X(:,:,3,1);
    invX(:,:,2,3) = X(:,:,1,3).*X(:,:,2,1) - X(:,:,1,1).*X(:,:,2,3);
    invX(:,:,3,1) = X(:,:,2,1).*X(:,:,3,2) - X(:,:,2,2).*X(:,:,3,1);
    invX(:,:,3,2) = X(:,:,1,2).*X(:,:,3,1) - X(:,:,1,1).*X(:,:,3,2);
    invX(:,:,3,3) = X(:,:,1,1).*X(:,:,2,2) - X(:,:,1,2).*X(:,:,2,1);
    invX = invX./detX; % using implicit expanion
%    invX = bsxfun(@rdivide, invX, detX); % for prior R2016b
elseif M == 4
    invX = zeros(I,J,M,M);
    detX = X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,4) + X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,2) + X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,3) + X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,3) + X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,4) + X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,1) + X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,4) + X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,1) + X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,2) + X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,2) + X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,3) + X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,3) - X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,4) - X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,2) - X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,4) - X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,1) - X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,3) - X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,2) - X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,4) - X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,3) - X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,2);
    invX(:,:,1,1) = X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,4) + X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,2) + X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,3) - X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,3) - X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,4) - X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,2);
    invX(:,:,1,2) = X(:,:,1,2).*X(:,:,3,4).*X(:,:,4,3) + X(:,:,1,3).*X(:,:,3,2).*X(:,:,4,4) + X(:,:,1,4).*X(:,:,3,3).*X(:,:,4,2) - X(:,:,1,2).*X(:,:,3,3).*X(:,:,4,4) - X(:,:,1,3).*X(:,:,3,4).*X(:,:,4,2) - X(:,:,1,4).*X(:,:,3,2).*X(:,:,4,3);
    invX(:,:,1,3) = X(:,:,1,2).*X(:,:,2,3).*X(:,:,4,4) + X(:,:,1,3).*X(:,:,2,4).*X(:,:,4,2) + X(:,:,1,4).*X(:,:,2,2).*X(:,:,4,3) - X(:,:,1,2).*X(:,:,2,4).*X(:,:,4,3) - X(:,:,1,3).*X(:,:,2,2).*X(:,:,4,4) - X(:,:,1,4).*X(:,:,2,3).*X(:,:,4,2);
    invX(:,:,1,4) = X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,3) + X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,4) + X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,2) - X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,4) - X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,2) - X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,3);
    invX(:,:,2,1) = X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,3) + X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,4) + X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,1) - X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,4) - X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,1) - X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,3);
    invX(:,:,2,2) = X(:,:,1,1).*X(:,:,3,3).*X(:,:,4,4) + X(:,:,1,3).*X(:,:,3,4).*X(:,:,4,1) + X(:,:,1,4).*X(:,:,3,1).*X(:,:,4,3) - X(:,:,1,1).*X(:,:,3,4).*X(:,:,4,3) - X(:,:,1,3).*X(:,:,3,1).*X(:,:,4,4) - X(:,:,1,4).*X(:,:,3,3).*X(:,:,4,1);
    invX(:,:,2,3) = X(:,:,1,1).*X(:,:,2,4).*X(:,:,4,3) + X(:,:,1,3).*X(:,:,2,1).*X(:,:,4,4) + X(:,:,1,4).*X(:,:,2,3).*X(:,:,4,1) - X(:,:,1,1).*X(:,:,2,3).*X(:,:,4,4) - X(:,:,1,3).*X(:,:,2,4).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,2,1).*X(:,:,4,3);
    invX(:,:,2,4) = X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,4) + X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,1) + X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,3) - X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,3) - X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,4) - X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,1);
    invX(:,:,3,1) = X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,4) + X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,1) + X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,2) - X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,2) - X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,4) - X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,1);
    invX(:,:,3,2) = X(:,:,1,1).*X(:,:,3,4).*X(:,:,4,2) + X(:,:,1,2).*X(:,:,3,1).*X(:,:,4,4) + X(:,:,1,4).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,1,1).*X(:,:,3,2).*X(:,:,4,4) - X(:,:,1,2).*X(:,:,3,4).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,3,1).*X(:,:,4,2);
    invX(:,:,3,3) = X(:,:,1,1).*X(:,:,2,2).*X(:,:,4,4) + X(:,:,1,2).*X(:,:,2,4).*X(:,:,4,1) + X(:,:,1,4).*X(:,:,2,1).*X(:,:,4,2) - X(:,:,1,1).*X(:,:,2,4).*X(:,:,4,2) - X(:,:,1,2).*X(:,:,2,1).*X(:,:,4,4) - X(:,:,1,4).*X(:,:,2,2).*X(:,:,4,1);
    invX(:,:,3,4) = X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,2) + X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,4) + X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,1) - X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,4) - X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,1) - X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,2);
    invX(:,:,4,1) = X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,2) + X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,3) + X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,3) - X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,1) - X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,2);
    invX(:,:,4,2) = X(:,:,1,1).*X(:,:,3,2).*X(:,:,4,3) + X(:,:,1,2).*X(:,:,3,3).*X(:,:,4,1) + X(:,:,1,3).*X(:,:,3,1).*X(:,:,4,2) - X(:,:,1,1).*X(:,:,3,3).*X(:,:,4,2) - X(:,:,1,2).*X(:,:,3,1).*X(:,:,4,3) - X(:,:,1,3).*X(:,:,3,2).*X(:,:,4,1);
    invX(:,:,4,3) = X(:,:,1,1).*X(:,:,2,3).*X(:,:,4,2) + X(:,:,1,2).*X(:,:,2,1).*X(:,:,4,3) + X(:,:,1,3).*X(:,:,2,2).*X(:,:,4,1) - X(:,:,1,1).*X(:,:,2,2).*X(:,:,4,3) - X(:,:,1,2).*X(:,:,2,3).*X(:,:,4,1) - X(:,:,1,3).*X(:,:,2,1).*X(:,:,4,2);
    invX(:,:,4,4) = X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,3) + X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,1) + X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,2) - X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,2) - X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,3) - X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,1);
    invX = invX./detX; % using implicit expanion
%    invX = bsxfun(@rdivide, invX, detX); % for prior R2016b
else % slow
    X = reshape(permute(X, [3,4,1,2]), [M,M,I*J]); % M x M x IJ
    eyeM = eye(M);
    invX = zeros(M,M,I*J);
    parfor ij = 1:I*J
            invX(:,:,ij) = X(:,:,ij)\eyeM;
    end
    invX = permute(reshape(invX, [M,M,I,J]), [3,4,1,2]); % I x J x M x M
end
end

%%% Trace %%%
function [ trX ] = local_trace( X, I, J, M )
if M == 2
    trX = X(:,:,1,1) + X(:,:,2,2);
elseif M == 3
    trX = X(:,:,1,1) + X(:,:,2,2) + X(:,:,3,3);
elseif M == 4
    trX = X(:,:,1,1) + X(:,:,2,2) + X(:,:,3,3) + X(:,:,4,4);
else % slow
    X = reshape(permute(X, [3,4,1,2]), [M,M,I*J]); % M x M x IJ
    trX = zeros(I*J,1);
    parfor ij = 1:I*J
            trX(ij) = trace(X(:,:,ij));
    end
    trX = reshape(trX, [I,J]); % I x J
end
end

%%% Determinant %%%
function [ detX ] = local_det( X, I, J, M )
if M == 2
    detX = X(:,:,1,1).*X(:,:,2,2) - X(:,:,1,2).*X(:,:,2,1);
elseif M == 3
    detX = X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,3) + X(:,:,2,1).*X(:,:,3,2).*X(:,:,1,3) + X(:,:,3,1).*X(:,:,1,2).*X(:,:,2,3) - X(:,:,1,1).*X(:,:,3,2).*X(:,:,2,3) - X(:,:,3,1).*X(:,:,2,2).*X(:,:,1,3) - X(:,:,2,1).*X(:,:,1,2).*X(:,:,3,3);
elseif M == 4
    detX = X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,4) + X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,2) + X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,3) + X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,3) + X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,4) + X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,1) + X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,4) + X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,1) + X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,2) + X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,2) + X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,3) + X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,3) - X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,4) - X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,2) - X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,4) - X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,1) - X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,3) - X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,2) - X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,4) - X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,3) - X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,2);
else % slow
    X = reshape(permute(X, [3,4,1,2]), [M,M,I*J]); % M x M x IJ
    detX = zeros(I*J,1);
    parfor ij = 1:I*J
            detX(ij) = det(X(:,:,ij));
    end
    detX = reshape(detX, [I,J]); % I x J
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%