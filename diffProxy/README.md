# DiffProxy

From Node Graph Optimization Using Differentiable Proxies   
Yiwei Hu, Paul Guerrero, Miloš Hašan, Holly Rushmeier, Valentin Deschaintre   
In SIGGRAPH '22 Conference Processing, Vancouver, BC, Canada, 2022.   
https://github.com/yiwei-hu/DiffProxy

**List of changes**   
- Deletion of Substance Designer dependencies.   
- Adding of C-PPTBF parameters for StyleGan2 training.   
- Creation of optimization.py file to run stochastic gradient descent optimization.   


Sliced Wasserstein Loss code from A Sliced Wasserstein Loss for Neural Texture Synthesis   
Eric Heitz, Kenneth Vanhoey, Thomas Chambon, Laurent Belcour   
In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021   
https://github.com/tchambon/A-Sliced-Wasserstein-Loss-for-Neural-Texture-Synthesis

### Requirements
```bash 
pip install -r requirements.txt
```

CUDA toolkit 11.1 or later is also required.

### Run stochastic gradient descent optimization
```bash 
python optimization.py --input image.png --params params.txt
```

`image.png` is the target image that the optimization must match.   
`params.txt` is a text file with the C-PPTBF parameters with zoom estimated in Phase #1 to initialize the optimization.   
When the error decreases during the optimization, the current image will be outputted and the parameters will be written in a txt file log, saved in `results/image/`.

To run the optimization with our pre-trained model, download on [link] `vgg19.pth` for the Sliced Wasserstein Loss and `diffProxy.pkl` the StyleGan2 pre-trained model and put them in current directory.

