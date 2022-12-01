from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from torchvision import transforms
import torch

from datasets.data_io import *


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.0, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.kwargs = kwargs

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()


    def build_list(self):
        metas = []

        # with open(self.listfile) as f:
        #     scans = f.readlines()
        #     scans = [line.rstrip() for line in scans]
        scans = self.listfile

        # scans
        for scene in scans:
            pair_file = scene + "/cams/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) < self.nviews -1:
                        continue
                    metas.append((scene, ref_view, src_views))
        
        return metas


    def __len__(self):
        return len(self.metas)


    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])
        assert float(lines[11].split()[2]) == 128
        return intrinsics, extrinsics, depth_min, depth_interval


    def read_img(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    
    def motion_blur(self, img):
        max_kernel_size = 3
        # Either vertial, hozirontal or diagonal blur
        mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
        ksize = np.random.randint(0, (max_kernel_size + 1) / 2) * 2 + 1  # make sure is odd
        center = int((ksize - 1) / 2)
        kernel = np.zeros((ksize, ksize))
        if mode == 'h':
            kernel[center, :] = 1.
        elif mode == 'v':
            kernel[:, center] = 1.
        elif mode == 'diag_down':
            kernel = np.eye(ksize)
        elif mode == 'diag_up':
            kernel = np.flip(np.eye(ksize), 0)
        var = ksize * ksize / 16.
        grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
        gaussian = np.exp(-(np.square(grid - center) + np.square(grid.T - center)) / (2. * var))
        kernel *= gaussian
        kernel /= np.sum(kernel)
        img = cv2.filter2D(img, -1, kernel)
        return img


    def data_augmentation(self, img):
        img = transforms.ColorJitter(brightness=50/255, contrast=(0.3, 1.5), saturation=0, hue=0)(img)
        img = self.motion_blur(np.array(img, dtype=np.float32) / 255.)
        return img


    def read_depth_hr(self, filename):
        # read pfm depth file
        # w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        # print('depth_hr: {}'.format(depth_hr.shape))
        # depth_lr = self.prepare_img(depth_hr)
        depth_lr = depth_hr

        h, w = depth_lr.shape
        depth_lr_ms = {
            "stage1": cv2.resize(depth_lr, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_lr, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_lr,
        }
        return depth_lr_ms


    def get_mask_hr(self, depth_ms, depth_min, depth_max):
        depth1, depth2, depth3 = depth_ms["stage1"], depth_ms["stage2"], depth_ms["stage3"]
        mask1 = np.array((depth1 >= depth_min) & (depth1 <= depth_max), dtype=np.float32)
        mask2 = np.array((depth2 >= depth_min) & (depth2 <= depth_max), dtype=np.float32)
        mask3 = np.array((depth3 >= depth_min) & (depth3 <= depth_max), dtype=np.float32)
        mask_lr_ms = {
            "stage1": mask1,
            "stage2": mask2,
            "stage3": mask3
        }
        return mask_lr_ms


    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath,
                                        '{}/blended_images/{:0>8}.jpg'.format(scan, vid)) 
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            img = self.read_img(img_filename)

            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_filename_hr = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))

                depth_ms = self.read_depth_hr(depth_filename_hr)

                # get depth values
                # depth_max = depth_interval * self.ndepths + depth_min
                # depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)
                depth_max = depth_interval * 128 + depth_min
                depth_interval_new = (depth_max - depth_min) / self.ndepths
                depth_values = np.arange(depth_min, depth_max, depth_interval_new, dtype=np.float32)

                # mask = mask_read_ms
                mask = self.get_mask_hr(depth_ms, depth_min, depth_max)

            imgs.append(img)

        # all
        # imgs = torch.stack(imgs)
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        # ms proj_mats
        proj_matrices = np.stack(proj_matrices)
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
                "depth": depth_ms,
                "depth_values": depth_values,
                "mask": mask,
                "depth_min": depth_min,
                "depth_max": depth_max,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}
