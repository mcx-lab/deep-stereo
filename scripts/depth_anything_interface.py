import sys

import cv2
import torch
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# These functions aim to fit based on disparity map

def get_model(DEVICE, MODEL_PATH, model_type = "base", encoder='vitl', max_depth=20.0):
    if model_type == "base":
        sys.path.append("/Depth-Anything-V2")
        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        model = DepthAnythingV2(**model_configs[encoder])
    elif model_type == "metric":
        sys.path.append("/Depth-Anything-V2/metric_depth")

        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }

        model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    else:
        print("Invalid Model Type")
        return None

    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(DEVICE).eval()
    return model

def estimated_depth_model(x,a):
    return a * x 

def get_pred_depth(depth, est_depth, config, fit_model, maxfev=1000, verbose=False):
    depth_flatten = depth.flatten()
    est_depth_flatten = est_depth.flatten()
    est_depth_flatten = est_depth_flatten[depth_flatten>=config["camera"]["min_reliable_range"]]
    depth_flatten = depth_flatten[depth_flatten>=config["camera"]["min_reliable_range"]]
    est_depth_flatten = est_depth_flatten[depth_flatten<=config["camera"]["max_reliable_range"]]
    depth_flatten = depth_flatten[depth_flatten<=config["camera"]["max_reliable_range"]]

    popt, _ = curve_fit(fit_model, est_depth_flatten, depth_flatten, maxfev=maxfev)

    if verbose: print(popt)

    pred_depth = fit_model(est_depth, *popt)

    # TODO: Calcualte r2 
    #if verbose:
    #    coefficient_of_dermination = r2_score(depth_flatten, pred_depth)
    #    print(f"R2: {coefficient_of_dermination}")
    
    return pred_depth, popt
