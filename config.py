from mlcpl import dataset
import torchvision

train_dataset = dataset.MSCOCO('/Users/max/Documents/datasets/COCO', split='train').drop_labels_random(0.5)
valid_dataset = dataset.MSCOCO('/Users/max/Documents/datasets/COCO', split='valid')

image_size = (448, 448)

batch_size = 32
accum_step = 4
lr = 2e-4
epochs = 60
early_stopping = 10
ema = 0.999
weight_decay = 1e-4

# logicmix
probability = 0.5
num_samples = (2, 4) # [inclusive, exclusive)

# DP-ASL
gamma_p, gamma_n, m = 0, 4, 0.05
omega_p, omega_n, n = 4, 0, 0   # if they are set as the same as above variables, it becomes ordinal asymmetric loss

# Pseudo-labeling
thresholds = (-4, 4)
E = 20

# Conventional Augmentation
data_aug = (
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandAugment(interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR),
    # torchvision.transforms.LinearTransformation(),
)

device = 'mps'