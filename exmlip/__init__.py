import torch

# e3nn 0.4.4 stores its Wigner constants with python slices; allow them under weights_only loading
torch.serialization.add_safe_globals([slice])
