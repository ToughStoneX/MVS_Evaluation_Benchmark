import torch
import chamfer3D.dist_chamfer_3D, fscore
import os
from plyfile import PlyData, PlyElement
import numpy as np 
import json
import argparse

from txt_io import write_evaluation_results

# -----------------------------------------------------------------------------------------------------------
# Manually set the variables
# -----------------------------------------------------------------------------------------------------------
# prefix = 'mvsnet'
# gtasfm_pcs_path = "/home/admin/workspace/project/codes/robustmvs/casmvsnet/outputs/gtasfm/eval/"
# gtasfm_gts_path = "/home/admin/workspace/project/datasets/evaluation_gtasfm/Points/stl/"

parser = argparse.ArgumentParser(description="MVS Evaluation for BlendedMVS")
parser.add_argument('--prefix', type=str, default='mvsnet')
parser.add_argument('--pcs_path', type=str, default='./outputs/blendedmvs/eval/')
parser.add_argument('--gts_path', type=str, default='./evaluation_blendedmvs/Points/stl/')
args = parser.parse_args()

prefix = args.prefix
gtasfm_pcs_path = args.pcs_path
gtasfm_gts_path = args.gts_path


scans = [
    "20190127_154841",
    "20190127_160317",
    "20190127_160737",
    "20190127_161129",
    "20190127_161419",
    "20190127_161922",
    "20190127_162256",
    "20190127_162742",
    "20190127_163215",
    "20190127_163742",
    "20190127_164516",
    "20190127_164840",
    "20190127_165303",
    "20190127_165928",
    "20190127_171304",
    "20190221_174234",
    "20190221_174830",
    "20190221_175345"
]

threshold_dict = {
    "20190127_154841": 0.0003,
    "20190127_160317": 0.002,
    "20190127_160737": 0.008,
    "20190127_161129": 0.001,

    "20190127_161419": 0.0005,
    "20190127_161922": 0.002,
    "20190127_162256": 0.0002,
    "20190127_162742": 0.001,

    "20190127_163215": 0.030,
    "20190127_163742": 0.030,
    "20190127_164516": 0.001,
    "20190127_164840": 0.003,

    "20190127_165303": 0.005,
    "20190127_165928": 0.001,
    "20190127_171304": 0.005,
    "20190221_174234": 0.002,

    "20190221_174830": 0.001,
    "20190221_175345": 0.005
}

result_dict = {}
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
for i, scan in enumerate(scans):
    print(f'{i+1}/{len(scans)}: {scan}')

    # pc_filename = f"mvsnet{scan}_l3.ply"
    pc_filename = f"{prefix}{scan}_l3.ply"
    pc_file_path = os.path.join(gtasfm_pcs_path, pc_filename)
    print(f"Loading {pc_file_path}")
    pc = PlyData.read(pc_file_path)
    pc_x = np.array(pc['vertex']['x'])[:, np.newaxis]
    pc_y = np.array(pc['vertex']['y'])[:, np.newaxis]
    pc_z = np.array(pc['vertex']['z'])[:, np.newaxis]
    pc_np = np.concatenate([pc_x, pc_y, pc_z], axis=1)  # [N, 3]
    pc_t = torch.from_numpy(pc_np).cuda().unsqueeze(dim=0)

    gt_filename = f"stl{scan}_total.ply"
    gt_file_path = os.path.join(gtasfm_gts_path, gt_filename)
    print(f"Loading {gt_file_path}")
    gt = PlyData.read(gt_file_path)
    gt_x = np.array(gt['vertex']['x'])[:, np.newaxis]
    gt_y = np.array(gt['vertex']['y'])[:, np.newaxis]
    gt_z = np.array(gt['vertex']['z'])[:, np.newaxis]
    gt_np = np.concatenate([gt_x, gt_y, gt_z], axis=1)  # [N, 3]
    gt_t = torch.from_numpy(gt_np).cuda().unsqueeze(dim=0)

    with torch.no_grad():
        print("Computing chamfer distance...")
        dist1, dist2, idx1, idx2 = chamLoss(pc_t, gt_t)
        print(f"dist1: {dist1.min()} - {dist1.max()}")
        print(f"dist2: {dist2.min()} - {dist2.max()}")
        dist1_mean = dist1.mean()
        dist2_mean = dist2.mean()
        dist12_mean = (dist1_mean + dist2_mean) / 2
        print(f"dist1_mean: {dist1_mean}, dist2_mean: {dist2_mean}, dist12_mean: {dist12_mean}")
        result_dict[scan] = {
            'dist1': dist1_mean.cpu().data.numpy().tolist(), 
            'dist2': dist2_mean.cpu().data.numpy().tolist(), 
            'dist_mean': dist12_mean.cpu().data.numpy().tolist()
        }
        print("Computing fscore...")
        threshold = threshold_dict[scan]
        f_score, precision, recall = fscore.fscore(dist1, dist2, threshold)
        print(f"f_score: {f_score}, precision: {precision}, recall: {recall}")
        result_dict[scan]["f_score"] = f_score.cpu().data.numpy().tolist()
        result_dict[scan]["precision"] = precision.cpu().data.numpy().tolist()
        result_dict[scan]["recall"] = recall.cpu().data.numpy().tolist()

print("---------------------------------------------------")
print(result_dict)
print("---------------------------------------------------")

print("Saving json files to ./results_gtasfm.json ...")
json_str = json.dumps(result_dict, sort_keys=False, indent=4, separators=(',', ': '))
with open(os.path.join(os.getcwd(), 'results_gtasfm.json'), 'w') as f:
    f.write(json_str)

print("Saving txt files to ./results_gtasfm.txt ...")
write_evaluation_results('./results_gtasfm.txt', scans, result_dict)









