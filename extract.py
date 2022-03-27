import os
import numpy as np
import cv2 as cv
import colorsys
import glob
import sys
import time

from util import mmap_manager
from yolact import gen_mask

def calc_iou(flow, masks, bboxes, classes):
    # cv.imwrite('result/flow.png', flow)
    env_flow = flow

    # end if the image has no hand
    if next((f for f in classes if f == 0), None) == None:
        print('no hands')
        return

    ### Calclate environment flow ###
    for j in range(len(masks)):
        ### Regenerate mask (this code ill be replaced) ###
        black = (masks[j]*255).astype(np.uint8)
        if classes[j] == 0:
            hand = black
        obj = black

        ### clipping ###
        obj = cv.bitwise_not(obj)
        # cv.imwrite('masks2/mask_{}.png'.format(j), obj)
        env_flow = cv.bitwise_and(env_flow, env_flow, mask=obj)
        # cv.imwrite('result/env_flow.png', env_flow)

    # print(env_flow.shape)
    ave_bgr = np.average(env_flow[np.where(env_flow[:, :, 0] + env_flow[:, :, 1] + env_flow[:, :, 2] != 0)], axis=0)
    ave_rgb = np.flip(ave_bgr)
    # print(ave_rgb)

    ### Flow except hand ###
    flow_hsv = cv.cvtColor(flow, cv.COLOR_BGR2HSV)
    ave_hsv = colorsys.rgb_to_hsv(ave_rgb[0]/255, ave_rgb[1]/255, ave_rgb[2]/255)
    # print((ave_hsv[0]*179, ave_hsv[1]*255, ave_hsv[2]*255))

    flow_clip = cv.bitwise_not(cv.inRange(flow_hsv, (max(0, ave_hsv[0]*179 - 30), 0, 0), (min(255, ave_hsv[0]*179 + 30), 255, 255)))

    flow_rec_diff = cv.subtract(flow_clip, hand)
    # cv.imwrite('result/flow_clip.png', flow_clip)
    # cv.imwrite('result/flow_rec_diff.png', flow_rec_diff)

    with open('result/result_iou.txt', mode='w') as f:
        for j in range(len(masks)):
            black = (masks[j]*255).astype(np.uint8)
            # cv.imwrite('masks/mask_{}.png'.format(j), black)
            mask = black
            bbox = np.array([(bboxes[j][0], bboxes[j][1]), (bboxes[j][2], bboxes[j][1]), (bboxes[j][2], bboxes[j][3]), (bboxes[j][0], bboxes[j][3])])
            stencil = np.zeros(flow_rec_diff.shape).astype(flow_rec_diff.dtype)
            cv.fillPoly(stencil, [bbox], [255, 255, 255])
            result = cv.bitwise_and(flow_rec_diff, stencil)
            # cv.imwrite('masks/bboxes_{}.png'.format(j), result)
            intersection = np.logical_and(result, mask)
            union = np.logical_or(result, mask)
            iou_score = np.sum(intersection) / np.sum(union)
            f.write('{}, class: {}, iou: {}, bbox:{}/{}/{}/{}\n'.format(j, classes[j], iou_score, bboxes[j][0], bboxes[j][1], bboxes[j][2], bboxes[j][3]))
            if iou_score >= 0.2:
                print('{}, class: {}, iou: {}, bbox:{}/{}/{}/{}'.format(j, classes[j], iou_score, bboxes[j][0], bboxes[j][1], bboxes[j][2], bboxes[j][3]))

if __name__ == '__main__':
    os.makedirs('masks', exist_ok=True)
    os.makedirs('result', exist_ok=True)

    trained_model='yolact/weights/yolact_plus_resnet50_54_800000.pth'
    net = gen_mask.prepare_net(trained_model)

    shape = (384, 640, 3)
    img_size = np.prod(shape)
    mm_image_in = mmap_manager.mmapManager('./flow_img_in.dat', img_size)
    mm_image_out = mmap_manager.mmapManager('./flow_img_out.dat', img_size)
    mm_status = mmap_manager.mmapManager('./status.dat', 4)

    cnt = 0

    images = glob.glob(os.path.join(sys.path[0], 'demo-frames', '*.png')) + \
            glob.glob(os.path.join(sys.path[0], 'demo-frames', '*.jpg'))
    images = sorted(images)
    for imfile in images:
    # while cnt < 10:
        cnt += 1
        print(cnt)

        cam_image = cv.imread(imfile)

        mm_image_in.WriteImage(cam_image)
        if cnt == 1:
            print('first')
            mm_status.WriteString('frst')
        else:
            print('sent')
            mm_status.WriteString('sent')

        # generate mask
        masks, bboxes, classes = gen_mask.gen_mask(net, cam_image)
        while mm_status.ReadString() in ['sent', 'frst']:
            pass
        flow = mm_image_out.ReadImage()

        if cnt > 1:
            start_iou = time.perf_counter()
            calc_iou(flow, masks, bboxes, classes)
            stop_iou = time.perf_counter()
            print("Calculate IoU:", (stop_iou - start_iou) * 1000, "ms")
    mm_image_in.dispose()
    mm_image_out.dispose()
    mm_status.dispose()