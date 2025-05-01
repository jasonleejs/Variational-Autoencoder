import torchvision
import torch
import matplotlib.pyplot as plt
import csv
import os

img_size = 28
ds = torchvision.datasets.EMNIST('./data/', train=True, download=False, split='balanced',
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.Resize((img_size,img_size)),
                                          torchvision.transforms.ToTensor(), # creating tensors
                                          torchvision.transforms.Lambda(lambda x: x.transpose(1, 2)),
                                          torchvision.transforms.Lambda((lambda x: torch.flatten(x).to(device))), # Tensor([784])
                            ]))
label_mapping = ds.class_to_idx


# Useful Functions
def convert_c2i(c, label_mapping=label_mapping):
    return label_mapping[c]

def convert_i2c(i, label_mapping=label_mapping):
    idx_to_char = {i: c for c, i in label_mapping.items()}
    return idx_to_char[i]

def draw(x, title=None):
    """
    Visualize a 28x28 grayscale image stored as a PyTorch tensor of shape (1, 28, 28)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()  # Move to CPU if on GPU, detach from graph
        if x.ndim == 1:
            x = x.numpy().reshape((img_size, img_size))
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.squeeze(0)  # Convert from (1, 28, 28) to (28, 28)
    
    plt.imshow(x, cmap='gray')
    plt.title(title)
    plt.axis('off')
    

def log_training_session(params, fpath="vae_mlp_train_log.csv"):
    """
    Logs a dictionary of training parameters and results into a CSV file.

    Args:
        params (dict): Dictionary containing hyperparameters and results.
        file (str, optional): CSV file path. Defaults to "vae_mlp_train_log.csv".
    """

    file_exists = os.path.isfile(fpath)
    field_names = list(params.keys())

    with open(fpath, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        
        if not file_exists:
            writer.writeheader()
    
        writer.writerow(params)

    print(f"Logged training session to {fpath}")