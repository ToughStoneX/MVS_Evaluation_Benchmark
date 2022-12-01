from torch.utils.data import Dataset
import numpy as np
import os, cv2, time
from PIL import Image
from torchvision import transforms
import pickle
import torch

from datasets.data_io import *


s_h, s_w = 0, 0
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.max_h, self.max_w = kwargs["max_h"], kwargs["max_w"]
        self.fix_res = kwargs.get("fix_res", False)  #whether to fix the resolution of input image.
        self.fix_wh = False

        assert self.mode == "test"
        self.metas = self.build_list()


    def build_list(self):
        metas = []
        scans = self.listfile

        # scans 
        for scan in scans:
            colmap_folder = os.path.join(self.datapath, self.mode, scan, 'colmap', 'dense')
            for subname in os.listdir(colmap_folder):
                pair_file = os.path.join(colmap_folder, subname, 'pair.txt')
                # read the pair file
                with open(pair_file) as f:
                    num_viewpoint = int(f.readline())
                    for view_idx in range(num_viewpoint):                       
                        # the pait.txt in our preprocessed gta_sfm data utilizes indexes started from 1 rather than 0
                        # hence each index should minus 1
                        ref_view = int(f.readline().rstrip()) - 1
                        src_views = [int(x) - 1 for x in f.readline().rstrip().split()[1::2]]
                        metas.append((scan, subname, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas


    def __len__(self):
        return len(self.metas)


    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img


    def read_cam(self, pose_filename, K_filename):
        extrinsics = self.read_pkl_array(pose_filename)
        intrinsics = self.read_pkl_array(K_filename)
        return intrinsics, extrinsics


    def read_depth(self, filename):
        return self.read_pkl_array(filename)


    def read_pkl_array(self, filename):
        with open(filename, 'rb') as f:
            array = pickle.load(f)
        return array 


    def read_depth_ms(self, filename):
        # read pfm depth file
        # w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        depth = self.read_depth(filename)

        h, w = depth.shape
        depth_ms = {
            "stage1": cv2.resize(depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth,
        }
        return depth_ms


    def get_mask_ms(self, depth_ms, depth_min, depth_max):
        depth1, depth2, depth3 = depth_ms["stage1"], depth_ms["stage2"], depth_ms["stage3"]
        mask1 = np.float32((depth1 >= depth_min) * 1.0) * np.float32((depth1 <= depth_max) * 1.0)
        mask2 = np.float32((depth2 >= depth_min) * 1.0) * np.float32((depth2 <= depth_max) * 1.0)
        mask3 = np.float32((depth3 >= depth_min) * 1.0) * np.float32((depth3 <= depth_max) * 1.0)
        mask_lr_ms = {
            "stage1": mask1,
            "stage2": mask2,
            "stage3": mask3
        }
        return mask_lr_ms


    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=32):
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics


    def __getitem__(self, idx):
        global s_h, s_w
        meta = self.metas[idx]
        scan, subname, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        imgs_seg = []
        depth_values = None
        proj_matrices = []

        colmap_folder = os.path.join(self.datapath, self.mode, scan, 'colmap', 'dense', subname)

        for i, vid in enumerate(view_ids):

            img_filename = os.path.join(colmap_folder, "processed", "images", "{:08d}.png".format(vid))
            K_filename = os.path.join(colmap_folder, "processed", "K", "{:08d}.pkl".format(vid))
            pose_filename = os.path.join(colmap_folder, "processed", "pose", "{:08d}.pkl".format(vid))
            depth_filename = os.path.join(colmap_folder, "processed", "depth", "{:08d}.pkl".format(vid))

            img = self.read_img(img_filename)
            intrinsics, extrinsics = self.read_cam(pose_filename, K_filename)
            # NOTE: the extrinsics matrix provided in GTA-SFM dataset is the inverse form
            # please refer to: https://github.com/HKUST-Aerial-Robotics/Flow-Motion-Depth/issues/6
            extrinsics = np.linalg.inv(extrinsics)

            # scale input
            img, intrinsics = self.scale_mvs_input(img, intrinsics, self.max_w, self.max_h)

            if self.fix_res:
                # using the same standard height or width in entire scene.
                s_h, s_w = img.shape[:2]
                self.fix_res = False
                self.fix_wh = True

            if i == 0:
                if not self.fix_wh:
                    # using the same standard height or width in each nviews.
                    s_h, s_w = img.shape[:2]

            # resize to standard height or width
            c_h, c_w = img.shape[:2]
            if (c_h != s_h) or (c_w != s_w):
                scale_h = 1.0 * s_h / c_h
                scale_w = 1.0 * s_w / c_w
                img = cv2.resize(img, (s_w, s_h))
                intrinsics[0, :] *= scale_w
                intrinsics[1, :] *= scale_h

            imgs.append(img)
            # extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_ms = self.read_depth_ms(depth_filename)
                depth = depth_ms["stage3"]
                # NOTE: the minimum and maximum depth value is selected manually, because the original
                #     depth values vary too much to be covered by a cost volume effectively.
                depth_array = depth.reshape((-1))
                depth_median = np.median(depth_array)
                depth_min = depth_array.min()
                depth_max = 2 * depth_median - depth_min
                # depth_min, depth_max = depth.min(), depth.max()
                # depth_max = 40
                mask_ms = self.get_mask_ms(depth_ms, depth_min, depth_max)
                depth_interval = (depth_max - depth_min) / self.ndepths
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

        #all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        # ms proj_mats
        proj_matrices[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

        proj_matrices_ms = {
            "stage1": proj_matrices,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}