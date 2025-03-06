import torch


class Normalization():
    """
    This class is for normalizing the spectrograms batch by batch.
    The normalization used is min-max, two modes 'framewise' and 'imagewise' can be selected.
    In this paper, we found that 'imagewise' normalization works better than 'framewise'

    If framewise is used, then X must follow the shape of (B, F, T)
    """

    def __init__(self, min, max, mode='imagewise'):
        if mode == 'framewise':
            def normalize(x):
                size = x.shape
                # Finding max values for each frame
                x_max = x.max(1, keepdim=True)[0]
                x_min = x.min(1, keepdim=True)[0]
                # If there is a column with all zero, nan will occur
                x_std = (x-x_min)/(x_max-x_min)
                x_std[torch.isnan(x_std)] = 0  # Making nan to 0
                x_scaled = x_std * (max - min) + min
                return x_scaled
        elif mode == 'imagewise':
            def normalize(x):
                size = x.shape
                # x_max = x.view(size[0], size[1]*size[2]).max(1, keepdim=True)[0]
                # x_min = x.view(size[0], size[1]*size[2]).min(1, keepdim=True)[0]
                x_max = x.flatten(1).max(1, keepdim=True)[0]
                x_min = x.flatten(1).min(1, keepdim=True)[0]
                x_max = x_max.unsqueeze(1)  # Make it broadcastable
                x_min = x_min.unsqueeze(1)  # Make it broadcastable
                x_std = (x-x_min)/(x_max-x_min)
                x_scaled = x_std * (max - min) + min
                # if piano roll is empty, turn them to min
                x_scaled[torch.isnan(x_scaled)] = min
                return x_scaled
        else:
            print(f'please choose the correct mode')
        self.normalize = normalize

    def __call__(self, x):
        return self.normalize(x)


class LatentNormalization:
    """
    Normalization class specifically for DAC latent variables with shape (B, C, T)
    where B is batch size, C is number of latent channels (72 or 96), and T is time steps.
    """

    def __init__(self, min_val, max_val, mode='batchwise'):
        self.min_val = min_val
        self.max_val = max_val

        if mode == 'channelwise':
            def normalize(x):
                # x shape: (B, C, T)
                # Normalize each channel independently
                x_max = x.max(dim=2, keepdim=True)[0]  # (B, C, 1)
                x_min = x.min(dim=2, keepdim=True)[0]  # (B, C, 1)

                x_std = (x - x_min) / (x_max - x_min)
                x_scaled = x_std * (max_val - min_val) + min_val
                x_scaled[torch.isnan(x_scaled)] = min_val
                return x_scaled

        elif mode == 'batchwise':
            def normalize(x):
                # Normalize across both channel and time dimensions for each batch
                # x shape: (B, C, T) -> flatten to (B, C*T)
                x_max = x.flatten(1).max(dim=1, keepdim=True)[0]  # (B, 1)
                x_min = x.flatten(1).min(dim=1, keepdim=True)[0]  # (B, 1)

                # Reshape for broadcasting
                x_max = x_max.view(-1, 1, 1)  # (B, 1, 1)
                x_min = x_min.view(-1, 1, 1)  # (B, 1, 1)

                x_std = (x - x_min) / (x_max - x_min)
                x_scaled = x_std * (max_val - min_val) + min_val
                x_scaled[torch.isnan(x_scaled)] = min_val
                return x_scaled

        else:
            raise ValueError(f'Unrecognized normalization mode: {mode}')

        self.normalize = normalize

    def __call__(self, x):
        return self.normalize(x)
