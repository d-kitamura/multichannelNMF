function [sep,cost] = bss_multichannelNMF(mix,ns,nb,fftSize,shiftSize,it,drawConv)
%
% bss_multichannelNMF: Blind source separation based on multichannel NMF
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
%   [sep,cost] = bss_multichannelNMF(mix,ns,nb,fftSize,shiftSize,it,drawConv)
%
% [inputs]
%        mix: observed mixture (len x mic)
%         ns: number of sources (scalar)
%         nb: number of bases for all sources (scalar)
%    fftSize: window length in STFT (scalar)
%  shiftSize: shift length in STFT (scalar)
%         it: number of iterations (scalar)
%   drawConv: plot cost function values in each iteration or not (true or false)
%
% [outputs]
%        sep: estimated signals (length x channel x ns)
%       cost: convergence behavior of cost function in multichannel NMF (it+1 x 1)
%

delta = 10^(-12); % to avoid numerical conputational instability

% Short-time Fourier transform
[X, window] = STFT(mix,fftSize,shiftSize,'hamming');
signalScale = sum(mean(mean(abs(X).^2,3),2),1);
X = X./signalScale; % signal scaling
[I,J,M] = size(X); % fftSize/2+1 x time frames x mics

% Obtain time-frequency-wise spatial covariance matrices
XX = zeros(I,J,M,M);
x = permute(X,[3,1,2]); % M x I x J
for i = 1:I
    for j = 1:J
        XX(i,j,:,:) = x(:,i,j)*x(:,i,j)' + eye(M)*delta; % observed spatial covariance matrix in each time-frequency slot
    end
end

% Multichannel NMF
[Xhat,T,V,H,Z,cost] = multichannelNMF(XX,ns,nb,it,drawConv);

% Multichannel Wiener filtering
Y = zeros(I,J,M,ns);
Xhat = permute(Xhat,[3,4,1,2]); % M x M x I x J
for i = 1:I
    for j = 1:J
        for src = 1:ns
            ys = 0;
            for k = 1:nb
                ys = ys + Z(k,src)*T(i,k)*V(k,j);
            end
            Y(i,j,:,src) = ys * squeeze(H(i,src,:,:))/Xhat(:,:,i,j)*x(:,i,j); % M x 1
        end
    end
end

% Inverse STFT for each source
Y = Y.*signalScale; % signal rescaling
for src = 1:ns
    sep(:,:,src) = ISTFT(Y(:,:,:,src), shiftSize, window, size(mix,1));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%