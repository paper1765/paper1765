import os
import cv2
import copy
import time
import shutil
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from multiprocessing import Manager
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from network import MotionNet_v21

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--filelist', type=str, default='ManualFake_preview.npy', help='the filelist to test')
args = parser.parse_args()

gpu_ids = str(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
gpu_ids = gpu_ids[0]

train_size = (512, 512)
ORG_FRAMES = Manager().dict()
VIDEO_LENGTH = 3  # the number of frames for each video to test, default = 32
DROP = 1


class MyDataset(Dataset):
    def __init__(self, num=0, file='', choice='train'):
        self.num = num
        self.size = train_size
        self.choice = choice
        self.filelist = file
        self.path_root = 'data/'

    def __getitem__(self, idx):
        return self.load_item(idx % self.num)

    def __len__(self):
        return self.num

    def load_item(self, item_idx):
        global ORG_FRAMES
        # e.g. filelist: path_of_landmark, path_of_input_video, path_of_mask_video, frame_length
        # ['FaceForensics/fake/Deepfakes/landmark/000_003/Face000/Frame000000.npy',
        # 'FaceForensics/fake/Deepfakes/videos/000_003.mp4',
        # 'FaceForensics/fake/Deepfakes/masks/000_003.mp4', 50]
        item0, item1, item2, frame_len = self.filelist[item_idx]
        path_mark = self.path_root + item0
        path_video = self.path_root + item1
        path_frame_video = path_video.replace('/videos/', '/frame_videos/')[:-4]
        path_frame_mask = path_video.replace('/videos/', '/frame_masks/')[:-4]
        path_mask = self.path_root + item2 if item2 != '' else ''

        frame_begin = int(path_mark[-10:-4])

        img_list, mask_list, org_list, flow_list, fname_list = None, None, None, None, []

        mark = np.load(path_mark[:-10] + '%06d.npy' % frame_begin, allow_pickle=True)
        x1, y1, x2, y2 = mark[0]

        tmp_size = int((x2 - x1) * 0.2)
        x1 -= tmp_size
        x2 += tmp_size
        y1 -= tmp_size
        y2 += tmp_size
        x1 = max(0, x1)
        y1 = max(0, y1)

        scale = max(self.size[0] / (y2 - y1), self.size[1] / (x2 - x1))
        dx = x1 + ((x2 - x1) * scale - self.size[0]) // 2 / scale
        dy = y1 + ((y2 - y1) * scale - self.size[1]) // 2 / scale
        buffer, marks = None, None
        for idx in range(frame_begin, frame_begin + VIDEO_LENGTH):
            fname = path_mark[:-10] + '%06d.jpg' % idx
            fname_list.append(fname)

            img = cv2.imread(path_frame_video + '/Frame%06d.jpg' % idx)
            img = img[..., ::-1]
            Hg, Wg, _ = img.shape

            if self.choice == 'test' and idx < frame_begin + VIDEO_LENGTH - DROP:
                ORG_FRAMES.update({fname[:-23] + fname[-15:]: [img, np.zeros(img.shape, dtype=np.uint8), np.zeros(img.shape, dtype=np.uint8)]})

            flow_buffer = torch.zeros([1, self.size[0], self.size[1], 2]).float()
            buffer = flow_buffer if buffer is None else torch.cat([buffer, flow_buffer], dim=0)

            img = img[y1:y2, x1:x2, :]
            H, W, _ = img.shape

            if os.path.exists(path_frame_mask + '/Frame%06d.jpg' % idx):
                mask = cv2.imread(path_frame_mask + '/Frame%06d.jpg' % idx)
                Hm, Wm, _ = mask.shape
                if Hg != Hm or Wg != Wm:
                    mask = cv2.resize(mask, (Wg, Hg))
                mask = mask[y1:y2, x1:x2, :]
                mask[mask > 10] = 255
            else:
                mask = np.zeros([H, W, 3])

            mark = np.load(path_mark[:-10] + '%06d.npy' % idx, allow_pickle=True)[1]
            mark = torch.FloatTensor(mark).unsqueeze(0)
            mark.sub_(torch.FloatTensor([dx, dy]))
            mark.mul_(torch.FloatTensor([scale, scale]))
            marks = mark if marks is None else torch.cat([marks, mark], dim=0)

            # relative resize and then center crop
            factor = max(self.size[0] / H, self.size[1] / W)
            img = cv2.resize(img, (int(W * factor)+1, int(H * factor)+1))
            mask = cv2.resize(mask, (int(W * factor)+1, int(H * factor)+1))
            H, W, _ = img.shape
            nH_left, nH_right = (H - self.size[0]) // 2, (H - self.size[0]) // 2 + self.size[0]
            nW_left, nW_right = (W - self.size[1]) // 2, (W - self.size[1]) // 2 + self.size[1]
            img = img[nH_left:nH_right, nW_left:nW_right, :]
            mask = mask[nH_left:nH_right, nW_left:nW_right, :]

            img = img.astype('float') / 255.
            img = (img - 0.5) / 0.5
            img = torch.tensor(img).float().permute(2, 0, 1)
            img_list = img.unsqueeze(0) if img_list is None else torch.cat([img_list, img.unsqueeze(0)], dim=0)

            mask = thresholding(mask)
            mask = mask.astype('float') / 255.
            mask = torch.tensor(mask).float().permute(2, 0, 1)[:1, :, :]
            mask_list = mask.unsqueeze(0) if mask_list is None else torch.cat([mask_list, mask.unsqueeze(0)], dim=0)
        label = 0 if '/real/' in path_video else 1
        if path_mask:
            label = int(torch.max(mask_list))
        return img_list, mask_list, torch.tensor(label), buffer, marks, fname_list, torch.tensor([x1, y1, x2, y2])

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lr = 1e-4
        self.networks = MotionNet_v21()
        self.gen = self.networks.cuda()
        self.load()

    def process(self, x, frames, marks):

        cls_output, mask_output = self.gen(x, frames, marks)
        B, V, _, H, W = mask_output.shape

        mask_output = mask_output[:, :-DROP, :, :, :].contiguous()

        cls_output = F.softmax(cls_output, dim=1)[:, 1:2]
        mask_output = F.softmax(mask_output, dim=2)[:, :, 1:2, :, :]
        cls_output = cls_output.view(-1, 1)
        mask_output = mask_output.view(-1, 1, H, W)

        return cls_output, mask_output

    def forward(self, x, gt=None, Cg=None):
        return self.gen(x, gt, Cg)

    def load(self):
        pth_path = '%s_weights.pth' % self.networks.name
        self.gen.load_state_dict(torch.load(pth_path))
        print('Load weights from [%s]' % pth_path)


def Deepfake_Testing(model, test_npy='', mode='test', save_localization=False):
    test_file = np.load('data/' + test_npy)

    test_num = len(test_file)
    test_dataset = MyDataset(test_num, test_file, choice='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=4, num_workers=4, shuffle=False)
    model.eval()
    gen_losses, y_prob_list, y_pred_list, y_gt_list = [], [], [], []
    f1, iou, fpr, auc = [], [], [], []
    video_map, video_gt_map = {}, {}

    if save_localization:
        global ORG_FRAMES
        ORG_FRAMES = Manager().dict()
        path_out_org = 'res/%s_%s_%s_%s_frame_original/' % (mode, gpu_ids, model.networks.name, test_npy[:-4])
        rm_and_make_dir(path_out_org)

    for cnt, items in enumerate(test_loader):
        print('Testing (%d/%d)' % (cnt, len(test_loader)), end='\r')
        Ii, Mg, Cg, buffer, marks = (item.cuda() for item in items[:-2])
        filename = items[-2]
        bbox_list = items[-1]
        tmp_shift = DROP
        tmp_file = []
        for bs_idx in range(len(filename[0])):
            for frame_idx in range(len(filename) - tmp_shift):
                tmp_file.append(filename[frame_idx][bs_idx])
        filename = tmp_file

        with torch.no_grad():
            Co, Mo = model.process(Ii, buffer, marks)

        Ii = Ii[:, :-DROP, :, :, :].contiguous()
        Mg = Mg[:, :-DROP, :, :, :].contiguous()

        _, _, _, H, W = Ii.shape
        Ii = Ii.view(-1, 3, H, W)
        Mg = Mg.view(-1, 1, H, W)
        Cg = Cg.view(-1, 1)
        Ii, Mg, Mo = convert(Ii, 127.5, 127.5), convert(Mg), convert(Mo)

        f1_list, iou_list, fpr_list, _, _ = metric(copy.deepcopy(Mo), Mg)

        Co = list(Co.cpu().detach().numpy())
        Cg = list(Cg.cpu().detach().numpy())

        for i in range(len(filename)):
            k = filename[i][:-20]
            if k in video_map.keys():
                tmp_f1, tmp_iou, tmp_fpr, tmp_Co = video_map[k]
                tmp_f1.append(f1_list[i])
                tmp_iou.append(iou_list[i])
                tmp_fpr.append(fpr_list[i])
                tmp_Co.append(Co[i // max(VIDEO_LENGTH-1, 1)])
                v = [tmp_f1, tmp_iou, tmp_fpr, tmp_Co]
            else:
                v = [[f1_list[i]], [iou_list[i]], [fpr_list[i]], [Co[i // max(VIDEO_LENGTH-1, 1)]]]
            video_map.update({k: v})

            if k in video_gt_map.keys():
                v = video_gt_map[k]
                v = max(v, Cg[i // max(VIDEO_LENGTH-1, 1)])
            else:
                v = Cg[i // max(VIDEO_LENGTH-1, 1)]
            video_gt_map.update({k: v})

            if save_localization:
                k = filename[i][:-23] + filename[i][-15:]
                x1, y1, x2, y2 = bbox_list[i // max(VIDEO_LENGTH-1, 1)].numpy()
                v_org, v_gt, v_res = ORG_FRAMES[k]
                Ho, Wo, _ = v_org.shape
                x1, x2 = np.clip([x1, x2], 0, Wo)
                y1, y2 = np.clip([y1, y2], 0, Ho)
                if int(abs(x2-x1)) == 0 or int(abs(y2-y1)) == 0:
                    continue
                tmp_gt = np.concatenate([Mg[i][:, :, 0:1], Mg[i][:, :, 0:1], Mg[i][:, :, 0:1]], axis=2).astype(np.uint8)
                tmp_gt = cv2.resize(tmp_gt, (int(abs(x2-x1)), int(abs(y2-y1))))
                tmp_gt = thresholding(tmp_gt)
                tmp_res = np.concatenate([Mo[i][:, :, 0:1], Mo[i][:, :, 0:1], Mo[i][:, :, 0:1]], axis=2).astype(np.uint8)
                tmp_res = cv2.resize(tmp_res, (int(abs(x2-x1)), int(abs(y2-y1))))
                v_gt[y1:y2, x1:x2, :] = tmp_gt
                v_res[y1:y2, x1:x2, :] = tmp_res
                tmp_v1, tmp_v2, tmp_v3 = ORG_FRAMES[k]
                tmp_v2[v_gt == 255] = 255
                tmp_v3 = np.mean(tmp_v3 + v_res, axis=2, keepdims=True)
                tmp_v3 = np.concatenate([tmp_v3, tmp_v3, tmp_v3], axis=2).astype(np.uint8)
                v = [tmp_v1, tmp_v2, tmp_v3]
                ORG_FRAMES.update({k: v})

    for k in sorted(video_gt_map.keys()):
        tmp_f1, tmp_iou, tmp_fpr, tmp_Co = video_map[k]
        tmp_f1, tmp_iou, tmp_fpr = np.mean(tmp_f1), np.mean(tmp_iou), np.mean(tmp_fpr)

        if np.max(tmp_Co) > 0.5:
            tmp_Co = np.array(tmp_Co)
            tmp_Co[tmp_Co <= 0.5] = 0
            tmp_Co = np.sum(tmp_Co) / np.count_nonzero(tmp_Co)
        else:
            tmp_Co = np.mean(tmp_Co)

        f1.append(tmp_f1)
        iou.append(tmp_iou)
        fpr.append(tmp_fpr)
        y_prob_list.append(tmp_Co)
        y_pred_list.append(0 if tmp_Co <= 0.5 else 1)
        y_gt_list.append(video_gt_map[k])

    if save_localization:
        save_name, save_frame, original_gt, original_pred = [], [], [], []
        for k in sorted(ORG_FRAMES.keys()):
            tmp_v1, tmp_v2, tmp_v3 = ORG_FRAMES[k]
            original_gt.append(tmp_v2)
            original_pred.append(tmp_v3)
            H, W, _ = tmp_v1.shape
            name = k.replace('/', '_')[:-4] + '.jpg'
            rtn_list = [tmp_v1, tmp_v2, thresholding(tmp_v3)]
            rtn = np.ones([H, W * len(rtn_list) + 10 * (len(rtn_list) - 1), 3], dtype=np.uint8) * 255
            for idx, tmp in enumerate(rtn_list):
                rtn[:H, W * idx + 10 * idx:W * (idx + 1) + 10 * idx, :] = tmp[..., ::-1]
            save_name.append(name)
            save_frame.append(rtn)
            ORG_FRAMES.pop(k)
        f1, iou, fpr, auc, skip = metric(original_pred, original_gt, isSkipReal=True, isSkipAUC=False)
        for name, rtn, sp in zip(save_name, save_frame, skip):
            if sp != 1:
                cv2.imwrite('%s/Localization_%s.jpg' % (path_out_org, name[:-4]), rtn)

    Pixel_AUC = np.mean(auc) if auc else 0
    Pixel_F1 = np.mean(f1)
    Pixel_IOU = np.mean(iou)
    Pixel_FPR = np.mean(fpr)
    Video_Acc = accuracy_score(y_pred=y_pred_list, y_true=y_gt_list, normalize=True)
    Video_F1 = f1_score(y_pred=y_pred_list, y_true=y_gt_list, average='macro')
    Video_Recall = 1 if np.min(y_gt_list) == np.max(y_gt_list) else recall_score(y_pred=y_pred_list, y_true=y_gt_list, average='macro')
    Video_AUC = 0 if np.min(y_gt_list) == np.max(y_gt_list) else roc_auc_score(y_true=y_gt_list, y_score=y_prob_list, average='macro')
    print('\nEvaluation for [%s]:' % test_npy)
    print('Video-level F1:%5.4f, Acc:%5.4f, Rec:%5.4f, AUC:%5.4f' % (Video_F1, Video_Acc, Video_Recall, Video_AUC))
    if save_localization:
        print('Pixel-level F1:%5.4f, IOU:%5.4f, FPR:%5.4f, AUC:%5.4f\n' % (Pixel_F1, Pixel_IOU, Pixel_FPR, Pixel_AUC))
        print('Note: if no ground-truth masks provided, the pixel-level metric is incorrect.\n')


def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def convert(x, num1=255., num2=0.0):
    x = x * num1 + num2
    return x.permute(0, 2, 3, 1).cpu().detach().numpy()


def thresholding(x, thres=0.5):
    x[x <= int(thres * 255)] = 0
    x[x > int(thres * 255)] = 255
    return x


def metric(premask_org, groundtruth_org, isSkipReal=False, isSkipAUC=True):
    f1_list, iou_list, fpr_list, auc_list, skip_list = [], [], [], [], []
    for idx in range(len(premask_org)):
        premask, groundtruth = premask_org[idx], groundtruth_org[idx]
        if not isSkipAUC and np.max(groundtruth) != np.min(groundtruth):
            auc_list.append(roc_auc_score((groundtruth[:, :, :1].reshape(-1)).astype('int'), premask[:, :, :1].reshape(-1)))
        premask = thresholding(premask)
        seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
        true_pos = float(np.logical_and(premask, groundtruth).sum())
        true_neg = np.logical_and(seg_inv, gt_inv).sum()
        false_pos = np.logical_and(premask, gt_inv).sum()
        false_neg = np.logical_and(seg_inv, groundtruth).sum()
        f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
        cross = np.logical_and(premask, groundtruth)
        union = np.logical_or(premask, groundtruth)
        iou = np.sum(cross) / (np.sum(union) + 1e-6)
        fpr = false_pos / (false_pos + true_neg + 1e-6)
        if isSkipReal and (np.max(groundtruth) == np.min(groundtruth)):
            skip_list.append(1)
            continue
        else:
            skip_list.append(0)
        if np.sum(cross) + np.sum(union) == 0:
            iou = 1
            f1 = 1
        f1_list.append(f1)
        iou_list.append(iou)
        fpr_list.append(fpr)
    auc_list = [0] if not auc_list else auc_list
    return f1_list, iou_list, fpr_list, auc_list, skip_list


if __name__ == '__main__':
    model = MyModel().cuda()
    Deepfake_Testing(model=model, test_npy=args.filelist, save_localization=True)
