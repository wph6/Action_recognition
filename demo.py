import argparse
import time
import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names, cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


def helpwaring(img):
    cv2.putText(img, '{}'.format('help'), (120, 350),
                cv2.FONT_HERSHEY_TRIPLEX, 8, (0, 0, 255), 20)


def fallwaring(img):
    cv2.putText(img, '{}'.format('fall'), (120, 350),
                cv2.FONT_HERSHEY_TRIPLEX, 8, (0, 0, 255), 20)


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider, height_size, cpu, track, smooth):
    helpnum = 15  # 20帧
    fallnum = 15

    waitnum = 0

    helptag = False  # 动作开始

    falling = 1
    x0 = y0 = x1 = y1 = 0
    falltag = False

    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./outfall.avi', fourcc, 20, (800,600))
    for img in image_provider:
        img = cv2.resize(img, (800, 600))
        fallflag1 = fallflag2 = False

        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)  # 人的关键点索引 后两维是个人得分paf均值 数量，所有点
        for kpt_id in range(all_keypoints.shape[0]):  # 依次将每个关节点信息缩放回原始图像上
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            t1 = time.time()
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

            # 连续动作
            helpaction = (pose_keypoints[7][1] < pose_keypoints[6][1] < pose_keypoints[15][1]
                          and
                          pose_keypoints[4][1] < pose_keypoints[3][1] < pose_keypoints[14][1])  # 举手过程

            # if (pose_keypoints[8][1] !=-1 and pose_keypoints[10][1] !=-1 and pose_keypoints[11][1] !=-1 and pose_keypoints[13][1] !=-1):
            #     fallflag1 = ( pose_keypoints[10][1] < pose_keypoints[8][1] and pose_keypoints[13][1] < pose_keypoints[11][1] )  # 脚高于腰
            if (pose_keypoints[2][1] != -1 and pose_keypoints[5][1] != -1 and pose_keypoints[9][1] != -1 and pose_keypoints[12][1] != -1 ):
                fallflag2 = ( pose_keypoints[9][1] < pose_keypoints[2][1] or pose_keypoints[12][1] < pose_keypoints[5][1])  # 膝盖高于肩膀  跌倒状态

            if waitnum % 5 == 0:  # help
                if waitnum / 5 % 2 != 0:  # 奇数次
                    if helpaction:
                        helptag = True  # 动作开始
                else:  # 偶次
                    if helptag:
                        if helpaction:
                            helpnum = 0
                        else:
                            helptag = False

            dx = (abs(pose_keypoints[13][0] - pose_keypoints[11][0]) + abs(
                pose_keypoints[10][0] - pose_keypoints[8][0])) / 2
            dy = (pose_keypoints[13][1] - pose_keypoints[11][1] + pose_keypoints[10][1] - pose_keypoints[8][1]) / 2

            if waitnum % 6 == 0:  # fall 过程
                if falling == 1:
                    y0 = dy
                    x0 = dx
                if falling == 2:
                    y1 = dy
                    x1 = dx
                    if y1 < y0 and x1 > x0:
                        falltag = True
                if falling == 3:
                    y2 = dy
                    x2 = dx
                    if falltag:
                        if y2 < y1 and x2 > x1:
                            fallnum = 0
                        falltag = False
                    falling = 1
                falling = falling + 1

            if fallflag1 or fallflag2 :
                fallnum = 0

        # if track:
        #     track_poses(previous_poses, current_poses, smooth=smooth)
        #     previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)

        if helpnum < 15:
            helpwaring(img)
            helpnum = helpnum + 1
        if fallnum < 15:
            fallwaring(img)
            fallnum = fallnum + 1

        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        # for pose in current_poses:
        # cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
        #               (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
        # if track:
        #     cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
        #                 cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

        out.write(img)
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(10)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1

        waitnum = waitnum + 1
    out.release()

        # fps = (fps + 1 / (time.time() - t1)) / 2
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    path = "checkpoint_iter_370000.pth"
    # path = "checkpoint_epoch_85_AP_0.402.pth"
    parser.add_argument('--checkpoint-path', type=str, default=path, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')

    # parser.add_argument('--images', default='./2.png', help='path to input image(s)')
    # parser.add_argument('--video', type=str, default='', help='path to video file or camera id')  # image

    parser.add_argument('--images', default='', help='path to input image(s)')
    parser.add_argument('--video', type=str, default='./fall3.mp4', help='path to video file or camera id')

    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=0, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=0, help='smooth pose keypoints')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
    else:
        args.track = 0

    run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth)

    #  ['nose', 'neck',
    #  'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
    #  'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
    #  'r_eye', 'l_eye',
    #  'r_ear', 'l_ear']
