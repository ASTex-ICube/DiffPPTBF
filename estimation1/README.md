# Optimization Phase #1 

### Requirements
```bash 
pip install -r requirements.txt
```

### Predict C-PPTBF parameters with number of feature points
Prediction of C-PPTBF parameters including the number of feature points parameter instead of the zoom parameter, to be used for Phase #2.

```bash 
python predict_points.py --input image.png
```

`image.png` is the reconstructed C-PPTBF image for which we want to estimate the C-PPTBF parameters.   
`--min [number]` and `--max [number]` must be specified if you don't use our pre-trained model but a new model, in order to apply the correct normalization for the number of feature points parameter. At the beginning of the training of the new model you will have two values displayed `MIN NUMBER OF POINTS` and `MAX NUMBER OF POINTS`, indicating the minimum and maximum numbers of feature points used in the images of the training dataset, these values will have to be used for these parameters.   
It will output a txt file with the parameters in this order: `tiling type, jittering, number of feature points, alpha, normBlend, wsmooth, winfeat, aniso, sigcos, delta, larp`

To run the prediction with our pre-trained model, download on [[link](https://seafile.unistra.fr/d/4c57922791fc406581f3/)] `complete_withPoints.h5` and `alpha.h5` and put them in current directory.

### Predict C-PPTBF parameters with zoom
Prediction of C-PPTBF parameters including the zoom parameter, to be used for DiffProxy optimization.

```bash 
python predict.py --input image.png
```

`image.png` is the reconstructed C-PPTBF image for which we want to estimate the C-PPTBF parameters.   
It will output a txt file with the parameters in this order: `tiling type, jittering, zoom, alpha, normBlend, wsmooth, winfeat, aniso, sigcos, delta, larp`

To run the prediction with our pre-trained model, download on [[link](https://seafile.unistra.fr/d/4c57922791fc406581f3/)] `complete_withoutAlpha.h5` and `alpha.h5` and put them in current directory.

### Train a new model 

- To train a new model including all the C-PPTBF parameters with the number of feature points:

```bash 
python completeEstimation_withPoints.py
```

- To train a new model including all the C-PPTBF parameters with the zoom parameter:

```bash 
python completeEstimation_withoutAlpha.py.py
```

- To train a new model with only the alpha parameter:

```bash 
python estimationAlpha.py
```

All these scripts will output .h5 model files to be used for predictions.
