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
    cv.imwrite('result/flow.png', flow)
    env_flow = flow

    # end if the image has no hand
    if next((f for f in classes if f == 0), None) == None:
        print('no hands')
        return

    ### Calclate environment flow ###
    for j in range(len(masks)):
        ### Regenerate mask (this code ill be replaced) ###
        black = (masks[j]*255).astype(np.uint8)
        # black = np.zeros((384, 640), dtype=np.uint8)
        # solutions = np.argwhere(masks[j] != 0)
        # for px in solutions:
        #     black[px[0]][px[1]] = 255.0            
        if classes[j] == 0:
            hand = black
        obj = black

        ### clipping ###
        # stencil = np.zeros(obj.shape).astype(obj.dtype)
        # bbox = np.array([(bboxes[j][0], bboxes[j][1]), (bboxes[j][2], bboxes[j][1]), (bboxes[j][2], bboxes[j][3]), (bboxes[j][0], bboxes[j][3])])
        # cv2.fillPoly(stencil, [bbox], [255, 255, 255])
        # result = cv2.bitwise_and(obj, stencil)
        obj = cv.bitwise_not(obj)
        cv.imwrite('masks2/mask_{}.png'.format(j), obj)
        env_flow = cv.bitwise_and(env_flow, env_flow, mask=obj)
        # result_rev = cv2.bitwise_not(result)
        # env_flow = cv2.bitwise_and(env_flow, env_flow, mask=result_rev)

        cv.imwrite('result/env_flow.png', env_flow)

    # print(env_flow.shape)
    ave_bgr = np.average(env_flow[np.where(env_flow[:, :, 0] + env_flow[:, :, 1] + env_flow[:, :, 2] != 0)], axis=0)
    ave_rgb = np.flip(ave_bgr)
    print(ave_rgb)

    # print(len(env_flow[:, :, 0].flatten()))
    # # print(len(env_flow[np.where(env_flow[:, :, 0].flatten() != 0)]))
    # print(len(env_flow[:, :, 0].flatten()[np.where(env_flow[:, :, 0].flatten() != 0)]))
    # # print(np.average(env_flow, axis=0))


    ### Flow except hand ###
    flow_hsv = cv.cvtColor(flow, cv.COLOR_BGR2HSV)
    # print(flow_hsv[0, 0])
    ave_hsv = colorsys.rgb_to_hsv(ave_rgb[0]/255, ave_rgb[1]/255, ave_rgb[2]/255)
    print((ave_hsv[0]*179, ave_hsv[1]*255, ave_hsv[2]*255))

    flow_clip = cv.bitwise_not(cv.inRange(flow_hsv, (max(0, ave_hsv[0]*179 - 30), 0, 0), (min(255, ave_hsv[0]*179 + 30), 255, 255)))

    flow_rec_diff = cv.subtract(flow_clip, hand)
    cv.imwrite('result/flow_clip.png', flow_clip)
    cv.imwrite('result/flow_rec_diff.png', flow_rec_diff)

    with open('result/result_iou.txt', mode='w') as f:
        for j in range(len(masks)):
            # solutions = np.argwhere(masks[j] != 0)
            # black = np.zeros((384, 640), dtype=np.uint8)
            # for px in solutions:
            #     black[px[0]][px[1]] = 255.0
            black = (masks[j]*255).astype(np.uint8)
            # cv.imwrite('masks/mask_{}.png'.format(j), black)
            mask = black
            bbox = np.array([(bboxes[j][0], bboxes[j][1]), (bboxes[j][2], bboxes[j][1]), (bboxes[j][2], bboxes[j][3]), (bboxes[j][0], bboxes[j][3])])
            # print(j, bbox)
            stencil = np.zeros(flow_rec_diff.shape).astype(flow_rec_diff.dtype)
            cv.fillPoly(stencil, [bbox], [255, 255, 255])
            result = cv.bitwise_and(flow_rec_diff, stencil)
            # cv.imwrite('masks/bboxes_{}.png'.format(j), result)
            intersection = np.logical_and(result, mask)
            union = np.logical_or(result, mask)
            iou_score = np.sum(intersection) / np.sum(union)
            # print(j, iou_score)
            f.write('{}, class: {}, iou: {}, bbox:{}/{}/{}/{}\n'.format(j, classes[j], iou_score, bboxes[j][0], bboxes[j][1], bboxes[j][2], bboxes[j][3]))
            if iou_score >= 0.2:
                print('{}, class: {}, iou: {}, bbox:{}/{}/{}/{}'.format(j, classes[j], iou_score, bboxes[j][0], bboxes[j][1], bboxes[j][2], bboxes[j][3]))


    # for j in range(len(masks)):
    #     if classes != 0:
    #         solutions = np.argwhere(masks[j] != 0)
    #         blask = np.zeros((384, 640), dtype=int)
    #         for px in solutions:
    #             black[px[0]][px[1]] = 255.0

def gen_flow():
    flow = cv.imread('data/out_44_to_out_45.png')
    return flow

# def gen_mask():
#     masks = np.load('data/mask_data.npy')
#     bboxes = np.load('data/bbox_data.npy')
#     classes = np.load('data/class_data.npy')        
#     return masks, bboxes, classes

def load_models():
    print("loading")

def init():
    load_models()

if __name__ == '__main__':
    os.makedirs('masks', exist_ok=True)
    os.makedirs('result', exist_ok=True)

    init()
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
        # wait unti camera image arrives
        # if cnt == 1:
        #     cam_image = cv.imread('data/out_44.jpg')
        # if cnt == 2:
        #     cam_image = cv.imread('data/out_45.jpg')
        # if cnt == 3:
        #     cam_image = cv.imread('data/out_46.jpg')
        # if cnt == 4:
        #     cam_image = cv.imread('data/out_47.jpg')
        # if cnt == 5:
        #     cam_image = cv.imread('data/out_48.jpg')
        # if cnt == 6:
        #     cam_image = cv.imread('data/out_49.jpg')
        # if cnt == 7:
        #     cam_image = cv.imread('data/out_50.jpg')
        # if cnt == 8:
        #     cam_image = cv.imread('data/out_51.jpg')
        # if cnt == 9:
        #     cam_image = cv.imread('data/out_52.jpg')
        # if cnt == 10:
        #     cam_image = cv.imread('data/out_53.jpg')
                
        # flow = gen_flow()
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
        # cv.imwrite('tmpimage.png', flow)
    mm_image_in.dispose()
    mm_image_out.dispose()
    mm_status.dispose()