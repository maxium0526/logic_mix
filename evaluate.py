from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from mlcpl.helper import *
from pathlib import Path
import torchmetrics
from models import Model
import config
from train_eval_fn import *

def main(dataset_split):
    model_path = 'output/train/best.pth'
    log_dir = f'output/eval_{dataset_split}/'

    if dataset_split == 'valid':
        dataset = config.valid_dataset
    
    dataset.records = dataset.records[:100]

    dataset.transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
    ])

    num_categories = dataset.num_categories

    model = Model(num_categories)

    single_value_validation_metrics = {
        'mAP@C': torchmetrics.classification.MultilabelAveragePrecision(num_categories, average='macro', validate_args=False),
    }

    multi_value_validation_metrics = {
    }

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=0, shuffle=False)
    
    model = model.to(config.device)
    model.load_state_dict(torch.load(model_path))

    logger = MultiLogger(log_dir, tblog=False)

    # Valid Loop
    preds, targets = evaluate(model, dataloader, device=config.device)

    single_value_record = {}
    for name, metric in single_value_validation_metrics.items():
        result = metric(preds, targets).detach().numpy()
        single_value_record[name] = result
    logger.add(f'single', single_value_record)

    for name, metric in multi_value_validation_metrics.items():
        result = metric(preds, targets).detach().numpy()
        logger.add(f'{name}', result)

    logger.flush()

if __name__=='__main__':
    main('valid')