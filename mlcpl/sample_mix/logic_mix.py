import torch
import numpy as np

class LogicMix(torch.utils.data.Dataset):
    def __init__(self, dataset, probability=1, mix_num_samples=2):
        self.dataset = dataset
        self.probability = probability
        self.mix_num_samples = mix_num_samples
        self.num_categories = self.dataset.num_categories

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if np.random.rand() > self.probability:
            img, target = self.dataset[idx]
            return img, target, False

        if type(self.mix_num_samples) is int:
            num_samples = self.mix_num_samples
        elif type(self.mix_num_samples) is tuple:
            num_samples = np.random.randint(*self.mix_num_samples)

        indices = np.random.randint(len(self.dataset), size=(num_samples))
        indices[0] = idx

        samples = [self.dataset[i] for i in indices]

        image = mix_images([image for image, target in samples])
        target = mix_targets([target for image, target in samples])

        return image, target, True
    
def mix_images(images):
    images = torch.stack(images)
    new_image = torch.mean(images, 0)

    return new_image

def mix_targets(targets):
    targets = torch.stack(targets)

    #compute must positive
    t = torch.where(torch.isnan(targets), 0, targets)
    t = torch.sum(t, dim=0)
    must_positive = torch.sign(t)

    #compute must negative
    t = torch.where(targets == 0, 0, 1)
    t = torch.sum(t, dim=0)
    must_negative = torch.where(t==0, 1, 0)

    new_target = must_positive
    new_target = torch.where(must_negative == 1, -1, new_target)
    new_target = torch.where(new_target == 0, torch.nan, new_target)
    new_target = torch.where(new_target == -1, 0, new_target)

    return new_target