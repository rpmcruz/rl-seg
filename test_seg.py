import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--dataset', default='BDD10K_Binary')
parser.add_argument('--batchsize', type=int, default=2)
parser.add_argument('--imgsize', type=int, default=2048)
parser.add_argument('--nregions', type=int, default=8)
args = parser.parse_args()

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

########################### DATA AUG ###########################

assert args.imgsize % args.nregions == 0
region_size = args.imgsize // args.nregions

transform = A.Compose([
    A.Resize(args.imgsize, args.imgsize),
    A.Normalize(0, 1),
    ToTensorV2(),
])

############################# DATA #############################

ds = getattr(data, args.dataset)
ds = ds('/data/auto', 'test', transform)
ts = torch.utils.data.DataLoader(ds, args.batchsize, num_workers=4, pin_memory=True)

############################ MODEL ############################

model = torch.load(args.model, map_location=device)

############################ LOOP ############################

metric = MulticlassJaccardIndex(ds.num_classes).to(device)

def to_patches(x):
    k = x.shape[1]
    x = x.unfold(2, region_size, region_size).unfold(3, region_size, region_size)
    x = x.permute(0, 2, 3, 1, 4, 5)
    x = x.reshape(-1, k, region_size, region_size)
    return x

for d in tqdm(ts):
    images = d['image'].to(device)  # N x C x H x W
    masks = d['mask'].to(device).long()[:, None]  # N x 1 x H x W

    images = to_patches(images)
    masks = to_patches(masks)[:, 0]

    preds = model(images)['out']
    preds = torch.softmax(preds, 1).argmax(1)
    metric.update(preds, masks)

print(args.model, metric.compute().item())