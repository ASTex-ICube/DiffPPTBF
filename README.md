# DiffPPTBF

This is the project page of the paper:

Guillaume Baldi, [Rémi Allègre](https://igg.icube.unistra.fr/index.php/R%C3%A9mi_All%C3%A8gre), [Jean-Michel Dischler](https://dpt-info.di.unistra.fr/~dischler/). **Differentiable Point Process Texture Basis Functions for inverse procedural modeling of cellular stochastic structures**. Accepted for publication in Computers & Graphics.

<!---
### C-PPTBF parameters

- tiling: Tiling type
- jittering: Random perturbation applied on point positions &beta;
- zoom: Scaling factor to zoom/unzoom $s$
- points: Number of points $K$
- alpha: Rotation angle &alpha;
- normBlend: Interpolation coefficient between the 2 windows &omega;
- wsmooth: Degree of smoothing in cellular window $s_c$ 
- winfeat: Random perturbation applied on feature points locations &gamma; 
- aniso: Anisotropy, scaling transform on Gaussian kernel &rho;
- sigcos: Sigma applied of Gaussian kernel &sigma;
- delta: Orientation of Gaussian kernel &Phi; 
- larp: Anisotropy of cellular window, interpolation factor between rectangular and Voronoi cells &lambda;
--->

The complete source code and procedural representations of cellular stochastic structures will be made progressively available, as of the beginning of April 2023.

### Source code organization and requirements

The source code is split in 4 main directories:
- [pix2pix](pix2pix) is a fork the Pix2Pix original code, with some changes.
- [estimation1](estimation1) is our model model for Optimization Phase #1, as described in the paper.
- [estimation2](estimation2) is the code for Optimization Phase #2, containing the implementation of the differentiable C-PPTBF, gradient descent with error metric based on the SWD1D, and Basin Hopping iterations.
- [diffProxy](diffProxy) is a fork of the DiffProxy original code, with some changes.

The requirements are described within each directory.

### Pipeline

1) **Reconstruction of a fake C-PPTBF image**: in the `pix2pix` folder you can find the steps to reconstruct a fake C-PPTBF image, by training and testing your own trained model or testing by using our pre-trained model. It will output images of the reconstruction ending by `_fake_B.png` which will be used in the next steps.  
2) **Optimization Phase #1**: in the `estimation` folder you can find the steps to estimate the tiling type and the initial values of continuous parameters by prediction using our pre-trained model. It will output txt files containing all the parameter values to be used in the Phase #2 and for DiffProxy.   
3) **Optimization Phase #2**:    
4) **DiffProxy**: in the `diffProxy` folder you can find the steps to perform the optimziation with DiffProxy using our pre-trained StyleGan2 model and the Sliced Wasserstein Loss. It will output a txt file for the logs of the minimization of the error function and the image matching the input image.   
