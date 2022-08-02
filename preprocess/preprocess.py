import os
import cv2
import shutil
import tqdm
import numpy as np
import argparse
from threading import Thread
import torch
from Retinaface import Retinaface
from mobilefacenet import MobileFaceNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ManualFake_preview', help='dataset to preprocess')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
args = parser.parse_args()

gpu_ids = str(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

np.random.seed(666666)
torch.manual_seed(666666)
torch.cuda.manual_seed(666666)
torch.backends.cudnn.deterministic = True

detector = Retinaface.Retinaface()
map_location = lambda storage, loc: storage.cuda() if torch.cuda.is_available() else 'cpu'
model = MobileFaceNet([112, 112], 136)
checkpoint = torch.load('mobilefacenet_model_best.pth.tar', map_location=map_location)
model.load_state_dict(checkpoint['state_dict'])
model = model.eval()
model.cuda()


def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


class BBox(object):
    # bbox is a list of [left, right, top, bottom]
    def __init__(self, bbox):
        self.left = bbox[0]
        self.right = bbox[1]
        self.top = bbox[2]
        self.bottom = bbox[3]
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]
        self.h = bbox[3] - bbox[2]

    # scale to [0,1]
    def projectLandmark(self, landmark):
        landmark_= np.asarray(np.zeros(landmark.shape))
        for i, point in enumerate(landmark):
            landmark_[i] = ((point[0]-self.x)/self.w, (point[1]-self.y)/self.h)
        return landmark_

    # landmark of (5L, 2L) from [0,1] to real range
    def reprojectLandmark(self, landmark):
        landmark_= np.asarray(np.zeros(landmark.shape))
        for i, point in enumerate(landmark):
            x = point[0] * self.w + self.x
            y = point[1] * self.h + self.y
            landmark_[i] = (x, y)
        return landmark_


def generate_retina_landmark_and_extract_frames(base_path, video_list, max_frame=50):
    video_path = base_path + 'videos/'
    mask_path = base_path + 'masks/'
    for idx, file in enumerate(video_list):
        mask_flag = True if os.path.exists(mask_path + file) else False
        mark_path = base_path + 'landmark/' + file[:-4] + '/'
        rm_and_make_dir(mark_path)

        frame_path = base_path + 'frame_videos/' + file[:-4] + '/'
        rm_and_make_dir(frame_path)

        frame_mask_path = base_path + 'frame_masks/' + file[:-4] + '/'
        rm_and_make_dir(frame_mask_path)

        bbox_list = []
        land_list = []
        reader = cv2.VideoCapture(video_path + file)
        if mask_flag:
            reader_mask = cv2.VideoCapture(mask_path + file)
        frame_idx = 0
        error_cnt = 0
        while error_cnt < 20:
            flag1, img = reader.read()
            flag2 = True
            if mask_flag:
                flag2, mask = reader_mask.read()
            if not (flag1 and flag2):
                error_cnt += 1
                continue

            cv2.imwrite(frame_path + 'Frame%06d.jpg' % (frame_idx + error_cnt), img)
            if mask_flag:
                cv2.imwrite(frame_mask_path + 'Frame%06d.jpg' % (frame_idx + error_cnt), mask)

            faces = detector(img)
            bbox_list_backup = [None] * len(bbox_list)
            land_list_backup = [None] * len(land_list)
            height, width, _ = img.shape
            for face in faces:
                if face[4] < 0.9:
                    continue
                x1 = face[0]
                y1 = face[1]
                x2 = face[2]
                y2 = face[3]
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                if w < 32 or h < 32:
                    continue
                size = int(min([w, h])*1.2)
                cx = x1 + w//2
                cy = y1 + h//2
                x1 = cx - size//2
                x2 = x1 + size
                y1 = cy - size//2
                y2 = y1 + size

                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)
                new_bbox = list(map(int, [x1, x2, y1, y2]))
                new_bbox = BBox(new_bbox)
                cropped = img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
                if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                    cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
                cropped_face = cv2.resize(cropped, (112, 112))

                if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
                    continue
                test_face = cropped_face.copy()
                test_face = test_face/255.0
                test_face = test_face.transpose((2, 0, 1))
                test_face = test_face.reshape((1,) + test_face.shape)
                input = torch.from_numpy(test_face).float()
                input = torch.autograd.Variable(input)
                input = input.cuda()
                landmark = model(input)[0].cpu().data.numpy()
                landmark = landmark.reshape(-1, 2)
                landmark = new_bbox.reprojectLandmark(landmark)
                bbox = np.asanyarray([x1, y1, x2, y2]).astype('int')
                landmark = np.asanyarray(landmark).astype('int')

                flag = True
                for i in range(len(bbox_list)):
                    if bbox_list[i] is None:
                        continue
                    bx1, by1, bx2, by2 = bbox_list[i]
                    bw, bh = abs(bx2 - bx1), abs(by2 - by1)
                    w, h = abs(x2 - x1), abs(y2 - y1)
                    if abs(bx1 - x1) < bw // 2 and abs(bw - w) < bw // 2 and abs(bh - h) < bh // 2:
                        flag = False
                        bbox_list_backup[i] = bbox
                        land_list_backup[i] = landmark
                if flag:
                    bbox_list_backup.append(bbox)
                    land_list_backup.append(landmark)
            bbox_list = bbox_list_backup
            land_list = land_list_backup
            for i in range(len(bbox_list)):
                if bbox_list[i] is None:
                    continue
                bbox, landmark = bbox_list[i], land_list[i]
                res_path = mark_path + 'Face%03d/' % i
                if not os.path.exists(res_path):
                    os.mkdir(res_path)
                np.save(res_path + 'Frame%06d.npy' % (frame_idx + error_cnt), [bbox, landmark])
            frame_idx += 1
            if frame_idx >= max_frame:
                break


def thread_process(base_path):
    video_path = base_path + 'videos/'
    video_list = sorted(os.listdir(video_path))
    mask_path = base_path + 'masks/'
    mask_list = sorted(os.listdir(mask_path)) if os.path.exists(mask_path) else []
    for folder in ['landmark/', 'frame_videos/', 'frame_masks/']:
        path = base_path + folder
        if not os.path.exists(path):
            os.mkdir(path)

    print('Process Dataset: [%s] with [%d/%d] Videos/Masks' % (base_path, len(video_list), len(mask_list)))
    video_list = video_list

    cnt_thread = min(len(video_list), 4)
    cnt_totall = len(video_list)
    cnt_per_thread = cnt_totall // cnt_thread

    thread_list = []
    begin_list = list(range(0, cnt_totall + cnt_per_thread - 1, cnt_per_thread))
    for item in range(len(begin_list)):
        begin = begin_list[item]
        thread = Thread(target=generate_retina_landmark_and_extract_frames, args=(base_path, video_list[begin:begin+cnt_per_thread]))
        thread_list.append(thread)
        thread.start()

    for _item in thread_list:
        _item.join()
    return 0


def preprocess(dataset='ManualFake_preview'):
    # extract landmark, video/mask frames
    base_path_list = [
        '../data/%s/fake/' % dataset,
        '../data/%s/real/' % dataset,
    ]
    for base_path in base_path_list:
        thread_process(base_path)
        print('Finish Dataset:  [%s]' % base_path)


def generate_filelist(dataset='ManualFake_preview'):
    # example of filelist: path_of_landmark, path_of_input_video, path_of_mask_video, frame_length
    # ['FaceForensics/fake/Deepfakes/landmark/000_003/Face000/Frame000000.npy',
    # 'FaceForensics/fake/Deepfakes/videos/000_003.mp4',
    # 'FaceForensics/fake/Deepfakes/masks/000_003.mp4', 50]
    res = []

    path_root = '../data/'
    prefix_len = len(path_root)

    # list the dataset for generating filelist
    # e.g., [[the path of landmark, the path of mask frame]]
    path_list = [
        ['%s/fake/landmark/' % dataset, '%s/fake/frame_masks/' % dataset],
        ['%s/real/landmark/' % dataset, ''],
    ]

    for item in path_list:
        path_mark = path_root + item[0]
        path_mask = path_root + item[1]

        video_list = sorted(os.listdir(path_mark))
        for file in video_list:
            path2 = path_mark + file + '/'
            flist2 = sorted(os.listdir(path2))
            for file2 in flist2:
                path3 = path2 + file2 + '/'
                flist3 = sorted(os.listdir(path3))
                res1 = path3 + flist3[0]
                path_input_video = path_mark.replace('/landmark/', '/videos/') + file + '.mp4'
                res2 = path_input_video if os.path.exists(path_input_video) else ''
                path_mask_video = path_mask + file + '.mp4'
                res3 = path_mask_video if os.path.exists(path_mask_video) else ''
                res4 = len(flist3)
                if res4 < 50:
                    continue
                res.append((res1[prefix_len:], res2[prefix_len:], res3[prefix_len:], res4))

    save_name = '%s.npy' % dataset
    print('Saved file list in [flist/%s]' % save_name)
    np.save('../data/%s' % save_name, res)


if __name__ == '__main__':
    preprocess(args.dataset)
    generate_filelist(args.dataset)
