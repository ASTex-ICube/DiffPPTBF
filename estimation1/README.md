# Optimization Phase #1 

### Requirements
```bash 
pip install -r requirements.txt
```

### Predict C-PPTBF parameters with number of feature points
Prediction of C-PPTBF parameters including the number of feature points parameter instead of the zoom parameter, to be used for the Phase #2.

```bash 
python predict_points.py --input image.png
```

`--min [number]` and `--max [number]` must be specified if you don't use our pre-trained model but a new model, in this case you have to specify the values displayed during the training of the new model (MIN NUMBER OF POINTS and MAX NUMBER OF POINTS) in order to apply the correct normalization for the number of points parameter.   
`image.png` is the reconstructed C-PPTBF image for which we want to estimate the C-PPTBF parameters.   
It will output a txt file with the parameters in this order : `tiling type, jittering, number of feature points, alpha, normBlend, wsmooth, winfeat, aniso, sigcos, delta, larp`

To run the prediction with our pre-trained model, download on [link] `complete_withPoints.h5` and `alpha.h5` and put them in current directory.

### Predict C-PPTBF parameters with zoom
Prediction of C-PPTBF parameters including the zoom parameter, to be used for diffProxy optimization.

```bash 
python predict.py --input image.png
```

`image.png` is the reconstructed C-PPTBF image for which we want to estimate the C-PPTBF parameters.   
It will output a txt file with the parameters in the format : `tiling jittering zoom alpha normBlend wsmooth winfeat aniso sigcos delta larp`

To run the prediction with our pre-trained model, download on [link] `complete_withoutAlpha.h5` and `alpha.h5` and put them in current directory.



