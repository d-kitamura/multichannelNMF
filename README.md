# Multichannel Nonnegative Matrix Factorization (Multichannel NMF)

## About
Sample MATLAB script for multichannel nonnegative matrix factorization (multichannel NMF) and its application to blind audio source separation.

## Contents
- bss_eval [dir]:           open source (GPLv3 license) for evaluating audio source separation performance
- input [dir]:              includes test audio signals (reverberation time is around 300 ms)
- bss_multichannelNMF.m:    apply pre- and post-processing for blind source separation (STFT, multichannel NMF, multichannel Wiener filtering, and ISTFT)
- ISTFT.m:			    inverse short-time Fourier transform
- main.m:			    main script with parameter settings
- multichannelNMF.m:		function of multichannel NMF
- STFT.m:			short-time Fourier transform

## Usage Note
Although monotonic dcreasing of cost function value (i.e., convergence) of the iterative update algorithm in multichannel NMF is theoretically guaranteed, this implementation is not. This is because a small value (machine epsilon) is added to parameters after update for avoiding computational instability.

Implementation is quite heuristic and not readable. This is because the direct implementation of multichannel NMF is very slow. In particular, for loop should be avoided as much as possible in MATLAB. One example is to calculate time-frequency-wise inverse matrix. Straightforward implementation is just looping inv(A) function, but it is quite slow. To speed up this kind of calculation, in this script, the inverse matrix is directly described using a well-known determinant formula. Stupid, but it's fast.

## Original paper
Multichannel NMF was proposed in 
* H. Sawada, H. Kameoka, S. Araki, and N. Ueda,	"Multichannel extensions of non-negative matrix factorization with complex-valued data," IEEE Transactions on Audio, Speech, and Language Processing, vol. 21, no. 5, pp. 971-982, May 2013.

However, the following papers are also important for understanding the spatial covariance model assumed in multichannel NMF:
* N. Q. K. Duong, E. Vincent, and R. Gribonval, "Underdetermined reverberant audio source separation using a fullrank spatial covariance model," IEEE Transaction on Audio, Speech, and Language Processing, vol. 18, no. 7, pp. 1830–1840, 2010.
* A. Ozerov and C. Févotte, "Multichannel nonnegative matrix factorization in convolutive mixtures for audio source separation," IEEE Transaction on Audio, Speech, and Language, Processing, vol. 18, no. 3, pp. 550–563, 2010.

## See Also
* HP: http://d-kitamura.net