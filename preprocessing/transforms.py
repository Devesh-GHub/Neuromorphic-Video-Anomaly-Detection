from torchvision import transforms
from PIL import Image


def get_transform(image_size=128):
    """
    image_size: 128 or 64
    Returns transform pipeline
    """
    # Handle both single integer and tuple sizes
    if isinstance(image_size, (tuple, list)):
        size = image_size
    else:
        size = (image_size, image_size)
    
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),  # Converts to [0,1] (H, W, C) → (C, H, W), Converts PIL → PyTorch tensor
    ])
    return transform