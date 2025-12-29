# Responsible for discovering and organizing dataset paths.
# - Locates training and testing video folders.
# - Ensures dataset structure consistency.
# - Abstracts dataset path handling from training logic.

import os
import glob

def load_ucsd_ped2(dataset_root):
    """
    dataset_root = path to UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2 OR path to UCSDped2 directly
    returns:
        train_videos: list of train folders
        test_videos: list of test folders
    """
    # Check if dataset_root points directly to a ped directory
    if dataset_root.endswith(("UCSDped1", "UCSDped2")):
        ped_path = dataset_root
    else:
        # Assume it's the parent directory, look for UCSDped2
        ped_path = os.path.join(dataset_root, "UCSDped2")
    
    train_path = os.path.join(ped_path, "Train")   # adding Train to the path
    test_path = os.path.join(ped_path, "Test")    # adding Test to the path

    train_videos = sorted(glob.glob(os.path.join(train_path, "Train*")))   # sorting all folders in train 
    test_videos = sorted(glob.glob(os.path.join(test_path, "Test*")))      # sorting all folders in test

    return train_videos, test_videos  
