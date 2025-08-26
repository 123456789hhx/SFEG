import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.DataProcessing import *
from module.SFEG import SFEG
from utils.f1score import calculate_f_measure

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
to_pil = transforms.ToPILImage()

imgsz = 256
batchsz = 1

ImgPath = "dataset/DeepCrack/test-image/"
GTPath = "dataset/DeepCrack/test-label/"
resultsdir = "results/DeepCrack/"

TestingSet = RoadDataset(ImgPath, GTPath, imgsz, imgsz)
TestingLoader = DataLoader(TestingSet, batch_size=batchsz, shuffle=False)

Network = SFEG(embed_dim=[32, 64, 128, 256, 512], depth=[3, 3, 3, 3, 3])
Network.load_state_dict(torch.load('./weights/best_DeepCrack.pth', map_location=DEVICE))

if torch.cuda.is_available():
    Network = Network.to(DEVICE)

Network.eval()
exist = os.path.exists(resultsdir)
if not exist:
    os.makedirs(resultsdir)

all_pred_edges = []
all_gt_edges = []

with torch.no_grad():
    for idx, samples in enumerate(TestingLoader):
        img, gt = samples['image'], samples['mask']
        if torch.cuda.is_available():
            img = img.to(DEVICE)
            gt = gt.to(DEVICE)
        gt[gt > 0] = 1
        mask, _, _ = Network(img)

        pred = mask.clone()
        lab = gt.clone()
        mask = torch.sigmoid(mask)

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        pred_pil = to_pil(pred.squeeze(0).cpu())
        lab_pil = to_pil(lab.squeeze(0).cpu())
        # img_pil = to_pil(img.squeeze(0).cpu())

        pred_pil.save(os.path.join(resultsdir, f'img{idx+1}_pre.png'))
        lab_pil.save(os.path.join(resultsdir, f'img{idx+1}_lab.png'))
        # img_pil.save(os.path.join(resultsdir, f'img{idx+1}_rgb.png'))

        all_pred_edges.extend(mask.cpu().numpy())
        all_gt_edges.extend(gt.cpu().numpy())

f_max = 0
pscore = 0
rscore = 0
best_threshold = 0
for threshold in np.linspace(0, 1, num=101):
    f_total = 0
    p_total = 0
    r_total = 0
    for pred_edges, gt_edges in zip(all_pred_edges, all_gt_edges):
        precision, recall, f_measure, _ = calculate_f_measure(pred_edges, gt_edges, threshold)
        p_total += precision
        r_total += recall
        f_total += f_measure
    p_avg = p_total / len(all_pred_edges)
    r_avg = r_total / len(all_pred_edges)
    f_avg = f_total / len(all_pred_edges)
    if f_avg > f_max:
        f_max = f_avg
        pscore = p_avg
        rscore = r_avg
        best_threshold = threshold
ods_score = f_max
print(f"ODS_P: {pscore:.4f}")
print(f"ODS_R: {rscore:.4f}")
print(f"ODS_F1: {ods_score:.4f}")

ois_scores = []
p_scores = []
r_scores = []
for pred_edges, gt_edges in zip(all_pred_edges, all_gt_edges):
    f_max = 0
    pscore = 0
    rscore = 0
    for threshold in np.linspace(0, 1, num=101):
        precision, recall, f_measure, _ = calculate_f_measure(pred_edges, gt_edges, threshold)
        if f_measure > f_max:
            f_max = f_measure
            pscore = precision
            rscore = recall
    ois_scores.append(f_max)
    p_scores.append(pscore)
    r_scores.append(rscore)

ois_score = np.mean(ois_scores)
p_scores = np.mean(p_scores)
r_scores = np.mean(r_scores)
print(f"OIS_P: {p_scores:.4f}")
print(f"OIS_R: {r_scores:.4f}")
print(f"OIS_F1: {ois_score:.4f}")

print(f"Best_Threshold: {best_threshold:.4f}")

miou_scores = []
for pred_edges, gt_edges in zip(all_pred_edges, all_gt_edges):
    _, _, _, iou = calculate_f_measure(pred_edges, gt_edges, best_threshold)
    miou_scores.append(iou)
miou = np.mean(miou_scores)

print(f"mIoU: {miou:.4f}")