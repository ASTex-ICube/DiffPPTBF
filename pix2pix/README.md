# Pix2Pix C-PPTBF

From Image-to-Image Translation with Conditional Adversarial Networks   
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros. In CVPR 2017   
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

**List of changes**   
- Adding of L2 binary loss by thresholding the image outputted during the training of the model.   

### Requirements
```bash 
pip install -r requirements.txt
```

### Train Pix2Pix C-PPTBF on a dataset 
```bash 
python train.py --dataroot ./datasets/pptbf/ --name training --model pix2pix --direction BtoA --lambda_bin 50
```

`/datasets/pptbf/` must contains a directory named `train` of concatenated 512*256 images, with the C-PPTBF image on the left side and the structure map on the right side. The threshold value of the structure map must be indicated in the file names, with the first 2 decimal digits just before the image format. For example `pptbf25.png` for an image with a threshold value of 0.25.   
`--no_thresh` must be specified otherwise, but the L2 binary loss will not be applied.   
`--lambda_bin` is the weight for the L2 binary loss, not indicating it will train the model with only the L1 loss.

Other options such as the use of GPUs can be find in `/options/`.

### Test a trained model on a dataset
```bash 
python test.py --dataroot ./datasets/pptbf/ --name training --model pix2pix --direction BtoA --num_test N
```

`/datasets/pptbf/` must contains a directory named `test` of concatenated 512*256 images, with the real or the C-PPTBF image on the left side and the structure map on the right side.

`--num_test` is the number of images in the directory.

### Test pre-trained model on a dataset

To test our trained model, download on [[link](https://seafile.unistra.fr/d/4c57922791fc406581f3/)] `latest_net_D.pth` and `latest_net_G.pth`, put them in `/checkpoints/training/` and then run the above test command line. The results will be saved in `/results/training/`.

### Filter an image with Mean Curvature Filter

```bash 
python filter.py --input image.png --iterations 40
```

`image.png` is the binary image on which we want to apply the filter.    
`--iterations` is the number of iterations of the filter to apply on the image, 40 by default.    
It will output the filtered image in current directory.