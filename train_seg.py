import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('--dataset', default='BDD10K_Binary')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--imgsize', type=int, default=2048)
parser.add_argument('--nregions', type=int, default=8)
args = parser.parse_args()

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch, torchvision
from time import time
import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

########################### DATA AUG ###########################

assert args.imgsize % args.nregions == 0
region_size = args.imgsize // args.nregions

transform = A.Compose([
    A.Resize(args.imgsize, args.imgsize),
    A.RandomCrop(region_size, region_size),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(p=1),
    A.Normalize(0, 1),
    ToTensorV2(),
])

############################# DATA #############################

ds = getattr(data, args.dataset)
ds = ds('/data/auto', 'train', transform)
tr = torch.utils.data.DataLoader(ds, args.batchsize, True, num_workers=4, pin_memory=True)

############################ MODEL ############################

model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=ds.num_classes)
model.to(device)

############################ LOOP ############################

opt = torch.optim.Adam(model.parameters())
loss_function = torch.nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    tic = time()
    loss_avg = 0
    for d in tr:
        image = d['image'].to(device)
        mask = d['mask'].to(device).long()
        ypred = model(image)['out']
        loss = loss_function(ypred, mask)
        loss_avg += float(loss) / len(tr)
        opt.zero_grad()
        loss.backward()
        opt.step()  # w = w - eta*dloss/dw
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Loss: {loss_avg}')

torch.save(model.cpu(), args.output)