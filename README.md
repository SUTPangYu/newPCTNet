

## Requirement
* Python 3.7
* PyTorch 1.5.0
* torchvision
* numpy
* Pillow
* Cython
## Training
1.Set the path of training sets in config.py  
2.Run train.py
## Testing
1.Set the path of testing sets in config.py    
2.Run generate_salmap.py (can generate the predicted saliency maps)  
3.Run generate_visfeamaps.py (can visualize feature maps)  
4.Run test_metric_score.py (can evaluate the predicted saliency maps in terms of fmax,fmean,wfm,sm,em,mae). You also can use the toolkit released by us:https://github.com/lartpang/Py-SOD-VOS-EvalToolkit.

}

