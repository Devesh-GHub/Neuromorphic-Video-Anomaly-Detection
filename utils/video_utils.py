import cv2
import numpy as np

def compute_optical_flow(prev_frame, next_frame):
    """
    Computes dense optical flow using Farneback method.
    Inputs: two grayscale frames (H, W)
    Output: flow magnitude, flow angle
    """

    flow = cv2.calcOpticalFlowFarneback(
        prev_frame,       # previous frame means the earlier frame in time
        next_frame,
        None,             # None means that the function will allocate the output array itself
        pyr_scale=0.5,    # image scale (<1) to build pyramids for each image
        levels=3,         # number of pyramid layers including the initial image
        winsize=15,       # averaging window size
        iterations=3,     # number of iterations at each pyramid level
        poly_n=5,         # size of the pixel neighborhood used to find polynomial expansion in each pixel
        poly_sigma=1.2,   # standard deviation of the Gaussian that is used to smooth derivatives
        flags=0           # operation flags modifying the algorithm behavior
    )

    # Convert flow to magnitude and direction
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # Convert Cartesian to Polar coordinates, flow[..., 0] --> x component, flow[..., 1] --> y component 

    return mag, ang
