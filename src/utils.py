import numpy as np
import torch
import src.config as cfg

limbs_start = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
limbs_end   = [0, 0, 2, 3, 4, 2, 6, 7, 15, 9, 10, 15, 12, 13, 2]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Offset
y, x = torch.meshgrid((torch.arange(cfg.CELL_NUM, dtype=torch.float, device = device),
                       torch.arange(cfg.CELL_NUM, dtype=torch.float, device=device)))
x_offset = x.unsqueeze(2).repeat(1, 1, 16)
y_offset = y.unsqueeze(2).repeat(1, 1, 16)

# Transparent
alpha = 0.6
# For drawing limb
limbs1 = [1,  2, 2, 3, 4, 2, 6, 7, 15,  9, 10, 15, 12, 13]
limbs2 = [2, 15, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# Colors used to draw joint boxes and limbs
colors = np.array([(255, 0, 0), (255, 0, 0),
                   (0, 255, 0), (0, 255, 0), (0, 255, 0),
                   (255, 255, 0), (255, 255, 0), (255, 255, 0),
                   (0, 0, 255), (0, 0, 255), (0, 0, 255),
                   (255, 0, 255), (255, 0, 255), (255, 0, 255),
                   (255, 0, 0)]
                   ) / 255
