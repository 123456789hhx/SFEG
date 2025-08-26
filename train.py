import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.DataProcessing import *
from module.SFEG import SFEG
import time
from utils.f1score import calculate_f_measure
from loss.loss import structure_loss
from tqdm import tqdm

DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

imgsz = 256
batchsz = 6
lr = 0.001

ImgPath = "dataset/DeepCrack/train-image/"
GTPath = "dataset/DeepCrack/train-label/"

TrainingSet = RoadDataset(ImgPath, GTPath, imgsz, imgsz, augment=True)
TrainingLoader = DataLoader(TrainingSet, batch_size=batchsz, shuffle=True)

valImgPath = "dataset/DeepCrack/val-image/"
valGTPath = "dataset/DeepCrack/val-label/"

ValSet = RoadDataset(valImgPath, valGTPath, imgsz, imgsz)
ValLoader = DataLoader(ValSet, batch_size=batchsz, shuffle=False)

Network = SFEG(embed_dim=[32, 64, 128, 256, 512], depth=[3, 3, 3, 3, 3])

if torch.cuda.is_available():
    Network = Network.to(DEVICE)

optimizer = torch.optim.Adam(Network.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
start_time = time.time()

print('======================Begin Training!======================')
best_score = 0
for epoch in tqdm(range(100)):
    SLoss = 0.0
    Network.train()
    for idx, samples in tqdm(enumerate(TrainingLoader)):
        if torch.cuda.is_available():
            Img, GT = samples['image'], samples['mask']
            Img = Img.to(DEVICE)
            GT = GT.to(DEVICE)

        Mask1, Mask2, Mask3 = Network(Img)
        Mask1 = Mask1.float()
        Mask2 = Mask2.float()
        Mask3 = Mask3.float()
        GT[GT > 0] = 1
        GT = GT.float()
        STerm1 = structure_loss(Mask1, GT)
        STerm2 = structure_loss(Mask2, GT)
        STerm3 = structure_loss(Mask3, GT)
        STerm = STerm1 + 0.5 * STerm2 + 0.5 * STerm3
        SLoss = SLoss + STerm
        optimizer.zero_grad()
        STerm.backward()
        optimizer.step()

    scheduler.step()
    print('epoch=' + str(epoch + 1) + ', SLoss=' + str(SLoss.detach().cpu().numpy()))

    Network.eval()
    with torch.no_grad():
        all_pred_edges = []
        all_gt_edges = []
        for samples in ValLoader:
            img, gt = samples['image'], samples['mask']
            if torch.cuda.is_available():
                img = img.to(DEVICE)
                gt = gt.to(DEVICE)
            gt[gt > 0] = 1
            mask, _, _ = Network(img)
            all_pred_edges.extend(mask.cpu().numpy())
            all_gt_edges.extend(gt.cpu().numpy())

        f_max = 0
        best_threshold = 0
        for threshold in np.linspace(0, 1, num=101):
            f_total = 0
            for pred_edges, gt_edges in zip(all_pred_edges, all_gt_edges):
                precision, recall, f_measure, _ = calculate_f_measure(pred_edges, gt_edges, threshold)
                f_total += f_measure
            f_avg = f_total / len(all_pred_edges)
            if f_avg > f_max:
                f_max = f_avg
        ods_score = f_max
        print("Validation ODS_F1:", ods_score)

        ois_scores = []
        for pred_edges, gt_edges in zip(all_pred_edges, all_gt_edges):
            f_max = 0
            for threshold in np.linspace(0, 1, num=101):
                precision, recall, f_measure, _ = calculate_f_measure(pred_edges, gt_edges, threshold)
                if f_measure > f_max:
                    f_max = f_measure
            ois_scores.append(f_max)
        ois_score = np.mean(ois_scores)
        print("Validation OIS_F1:", ois_score)
        if ods_score + ois_score > best_score:
            best_score = ods_score + ois_score
            filename = f"./best_DeepCrack.pth"
            torch.save(Network.state_dict(), './' + filename)
            print('Model has been successfully saved in this epoch!')
    print('=============================================')

torch.save(Network.state_dict(), './best_DeepCrack_final.pth')
end_time = time.time()
running_time = end_time - start_time
print("Running time:", running_time, "seconds")
print("======================Training Finished!======================")