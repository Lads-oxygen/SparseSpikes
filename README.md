# Super-Resolution of Point-Wise Sources via the Beurling LASSO

This repository contains the implementation and experimental results for my Master's dissertation on the super-resolution of point-wise sources. The project focuses on recovering sparse discrete measures from blurred and noisy observations using the Beurling LASSO (BLASSO) framework.

The repository includes all scripts used to generate the figures in the final report and poster presentation, as well as reconstructions on real and simulated microscopy data.

## Dataset

The SMLM experiments use data from the publicly available [EPFL Single Molecule Localization Microscopy (SMLM) Challenge](https://srm.epfl.ch/) dataset.

## Acknowledgements

This project was supervised by Dr. Clarice Poon and draws from the following papers:
- Candes, E. J., & Fernandez-Granda, C. (2012). [Towards a Mathematical Theory of Super-Resolution](https://doi.org/10.1007/s10208-013-9152-x). Communications on Pure and Applied Mathematics, 67(6), 906–956.
- Duval, V., & Peyré, G. (2015). [Exact Support Recovery for Sparse Spikes Deconvolution](https://doi.org/10.1137/140977335). Foundations of Computational Mathematics, 15(5), 1315–1355.
- Denoyelle, Q., Duval, V., Peyré, G., & Soubies, E. (2018). [The Sliding Frank-Wolfe Algorithm and Its Application to Super-Resolution Microscopy](https://doi.org/10.1137/17M1131726). SIAM Journal on Imaging Sciences, 11(1), 1–47.
- Courbot, J.-B., & Colicchio, B. (2021). [Boosted Sliding Frank-Wolfe for Sparse Inverse Problems](https://doi.org/10.1088/1361-6420/abc804). Inverse Problems, 37(2), 025002.