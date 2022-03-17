import os
import numpy as np
import cv2
import colorsys

def calc_iou(flow, masks, bboxes, classes):
    env_flow = flow

    ### Calclte environment flow ###
    for j in range(len(masks)):
        ### Regenerate mask (this code ill be replaced) ###
        solutions = np.argwhere(masks[j] != 0)
        black = np.zeros((384, 640), dtype=np.uint8)
        for px in solutions:
            black[px[0]][px[1]] = 255.0            
        if classes[j] == 0:
            hand = black
        obj = black

        ### clipping ###
        # stencil = np.zeros(obj.shape).astype(obj.dtype)
        # bbox = np.array([(bboxes[j][0], bboxes[j][1]), (bboxes[j][2], bboxes[j][1]), (bboxes[j][2], bboxes[j][3]), (bboxes[j][0], bboxes[j][3])])
        # cv2.fillPoly(stencil, [bbox], [255, 255, 255])
        # result = cv2.bitwise_and(obj, stencil)
        obj = cv2.bitwise_not(obj)
        env_flow = cv2.bitwise_and(env_flow, env_flow, mask=obj)
        # result_rev = cv2.bitwise_not(result)
        # env_flow = cv2.bitwise_and(env_flow, env_flow, mask=result_rev)
        cv2.imwrite('result/env_flow.png', env_flow)

    print(env_flow.shape)
    ave_bgr = np.average(env_flow[np.where(env_flow[:, :, 0] + env_flow[:, :, 1] + env_flow[:, :, 2] != 0)], axis=0)
    ave_rgb = np.flip(ave_bgr)
    print(ave_rgb)

    # print(len(env_flow[:, :, 0].flatten()))
    # # print(len(env_flow[np.where(env_flow[:, :, 0].flatten() != 0)]))
    # print(len(env_flow[:, :, 0].flatten()[np.where(env_flow[:, :, 0].flatten() != 0)]))
    # # print(np.average(env_flow, axis=0))


    ### Flow except hand ###
    flow_hsv = cv2.cvtColor(flow, cv2.COLOR_BGR2HSV)
    # print(flow_hsv[0, 0])
    ave_hsv = colorsys.rgb_to_hsv(ave_rgb[0]/255, ave_rgb[1]/255, ave_rgb[2]/255)
    print(ave_hsv)
    flow_clip = cv2.bitwise_not(cv2.inRange(flow_hsv, (max(0, ave_hsv[0]*255 - 15), 0, 0), (min(255, ave_hsv[0]*255 + 15), 255, 255)))
    flow_rec_diff = cv2.subtract(flow_clip, hand)
    cv2.imwrite('result/flow_clip.png', flow_clip)
    # cv2.imwrite('result/flow_clip_item.png', cv2.subtract(flow_clip, hand))

    with open('result/result_iou.txt', mode='w') as f:
        for j in range(len(masks)):
            solutions = np.argwhere(masks[j] != 0)
            black = np.zeros((384, 640), dtype=np.uint8)
            for px in solutions:
                black[px[0]][px[1]] = 255.0            
            cv2.imwrite('masks/mask_{}.png'.format(j), black)
            mask = black
            bbox = np.array([(bboxes[j][0], bboxes[j][1]), (bboxes[j][2], bboxes[j][1]), (bboxes[j][2], bboxes[j][3]), (bboxes[j][0], bboxes[j][3])])
            # print(j, bbox)
            stencil = np.zeros(flow_rec_diff.shape).astype(flow_rec_diff.dtype)
            cv2.fillPoly(stencil, [bbox], [255, 255, 255])
            result = cv2.bitwise_and(flow_rec_diff, stencil)
            cv2.imwrite('masks/bboxes_{}.png'.format(j), result)
            intersection = np.logical_and(result, mask)
            union = np.logical_or(result, mask)
            iou_score = np.sum(intersection) / np.sum(union)
            # print(j, iou_score)
            f.write('{}, class: {}, iou: {}\n'.format(j, classes[j], iou_score))


    # for j in range(len(masks)):
    #     if classes != 0:
    #         solutions = np.argwhere(masks[j] != 0)
    #         blask = np.zeros((384, 640), dtype=int)
    #         for px in solutions:
    #             black[px[0]][px[1]] = 255.0

def gen_flow():
    flow = cv2.imread('data/out_44_to_out_45.png')
    return flow

def gen_mask():
    masks = np.load('data/mask_data.npy')
    bboxes = np.load('data/bbox_data.npy')
    classes = np.load('data/class_data.npy')        
    return masks, bboxes, classes

def load_models():
    print("loading")

def init():
    load_models()

if __name__ == '__main__':
    os.makedirs('masks', exist_ok=True)
    os.makedirs('result', exist_ok=True)

    init()

    flow = gen_flow()
    masks, bboxes, classes = gen_mask()
    calc_iou(flow, masks, bboxes, classes)