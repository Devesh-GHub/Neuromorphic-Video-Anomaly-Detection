from torch.utils.tensorboard import SummaryWriter         # SummaryWriter is the main TensorBoard interface in PyTorch 
import os


class Logger:
    def __init__(self, log_dir="runs/experiment"):        # where TensorBoard logs will be saved
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)              # everything logged via this writer will be saved in log_dir

    def log_loss(self, loss, step):                       # logs training loss
        self.writer.add_scalar("Loss/train", loss, step)  # Loss/train is graph name in TensorBoard, loss --> y-axis, step --> x-axis

    def log_lr(self, optimizer, step):                    # logs learning rate
        lr = optimizer.param_groups[0]["lr"]              # PyTorch stores optimizer parameters in parameter groups. Here, we access the learning rate of the first parameter group.
        self.writer.add_scalar("LearningRate", lr, step)  # LearningRate is graph name in TensorBoard, lr --> y-axis, step --> x-axis

    def log_images(self, original, reconstructed, step, max_images=4):  # logs original and reconstructed images
        """
        original, reconstructed: tensors (B, C, H, W)
        """
        self.writer.add_images("Original", original[:max_images], step)            # Original is graph name in TensorBoard, original[:max_images] selects first max_images images from the batch
        self.writer.add_images("Reconstructed", reconstructed[:max_images], step)  # Reconstructed is graph name in TensorBoard, reconstructed[:max_images] selects first max_images images from the batch

    def close(self):                                      # closes the SummaryWriter
        self.writer.close()                               # important to free up resources and ensure all data is written to disk
               