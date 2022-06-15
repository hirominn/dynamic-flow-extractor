import os
from shutil import move
import numpy as np
import cv2 as cv
import colorsys
import glob
import sys
import time
import socket
import math
from scipy.spatial import distance

from util import mmap_manager
from yolact import gen_mask, yolact

def calc_iou(flow, masks, bboxes, classes, isDebug):
    items = ''

    if(isDebug): cv.imwrite('result/flow.png', flow)
    env_flow = flow

    # end if the image has no hand
    if next((f for f in classes if f == 0), None) == None:
        print('no hands')
        return printItems(classes, bboxes)

    print(classes)

    obj_flows = []
    ### Calclate environment flow ###
    for j in range(len(masks)):
        ### Regenerate mask (this code ill be replaced) ###
        obj = (masks[j]*255).astype(np.uint8)
        obj_flow_rgb = np.flip(np.average(env_flow[np.where(obj[:, :] != 0)], axis=0))/255
        obj_flow_hsv = colorsys.rgb_to_hsv(obj_flow_rgb[0], obj_flow_rgb[1], obj_flow_rgb[2])
        if classes[j] == 0:
            hand = obj
            hand_flow_hsv = obj_flow_hsv
            hand_box = bboxes[j]
        # print(classes[j], obj_flow_hsv)
        obj_flows.append(obj_flow_hsv)

        ### clipping ###
        obj = cv.bitwise_not(obj)
        # cv.imwrite('masks2/mask_{}.png'.format(j), obj)
        env_flow = cv.bitwise_and(env_flow, env_flow, mask=obj)
    if(isDebug): cv.imwrite('result/env_flow.jpg', env_flow)

    # print(env_flow.shape)
    ave_bgr = np.average(env_flow[np.where(env_flow[:, :, 0] + env_flow[:, :, 1] + env_flow[:, :, 2] != 0)], axis=0)
    ave_rgb = np.flip(ave_bgr)

    ### Flow except hand ###
    flow_hsv = cv.cvtColor(flow, cv.COLOR_BGR2HSV)
    env_flow_hsv = cv.cvtColor(env_flow, cv.COLOR_BGR2HSV)
    ave_hsv = colorsys.rgb_to_hsv(ave_rgb[0]/255, ave_rgb[1]/255, ave_rgb[2]/255)

    # print((ave_hsv[0]*180, ave_hsv[1]*255, ave_hsv[2]*255))
    var_hue = np.var(env_flow_hsv[np.where(env_flow[:, :, 0] + env_flow[:, :, 1] + env_flow[:, :, 2] != 0)], axis=0)[0]
    var_sat = np.var(env_flow_hsv[np.where(env_flow[:, :, 0] + env_flow[:, :, 1] + env_flow[:, :, 2] != 0)], axis=0)[1]
    max_sat = np.max(flow_hsv[np.where(env_flow[:, :, 0] + env_flow[:, :, 1] + env_flow[:, :, 2] != 0)], axis=0)[1]
    print("var_hue", var_hue, "var_sat :", var_sat, "max_sat :", max_sat)

    def calc_euclid_distance_from_env(x):
        return distance.euclidean((x[1] * 255 * math.cos(math.radians(x[0] * 2*180)), x[1] * 255 * math.sin(math.radians(x[0] * 2*180))), (ave_hsv[1] * 255 * math.cos(math.radians(ave_hsv[0] * 2*180)), ave_hsv[1] * 255 * math.sin(math.radians(ave_hsv[0] * 2*180))))
    diff_from_env = list(map(calc_euclid_distance_from_env, obj_flows))
    # print("diff from env :\n", diff_from_env)

    def calc_euclid_distance_from_hand(x):
        return distance.euclidean((x[1] * 255 * math.cos(math.radians(x[0] * 2*180)), x[1] * 255 * math.sin(math.radians(x[0] * 2*180))), (hand_flow_hsv[1] * 255 * math.cos(math.radians(hand_flow_hsv[0] * 2*180)), hand_flow_hsv[1] * 255 * math.sin(math.radians(hand_flow_hsv[0] * 2*180))))
    diff_from_hand = list(map(calc_euclid_distance_from_hand, obj_flows))
    # print("diff from hand:\n", diff_from_hand)        

    for j in range(len(masks)):
        if(lenFromHand(bboxes[j], hand_box) < 0.20):
            handDist = "near"
        else:
            handDist = "far"
        if(diff_from_env[j] >= 7.5 and diff_from_hand[j] <= 4.0): 
            print('{}, hand_dist:{}'.format(class_names[classes[j]],lenFromHand(bboxes[j], hand_box)))
            print('{}, x[0]:{},x[1]:{},diff_env:{:0>4},diff_hand:{:0>4}\n'.format(class_names[classes[j]],obj_flows[j][0] * 180 * 2, obj_flows[j][1] * 255, diff_from_env[j], diff_from_hand[j]))
            movingRate = 1
        else :
            movingRate = 0
        items += '{},{},{}/{}/{}/{},{},{}\n'.format(class_names[classes[j]], classes[j], bboxes[j][0]/img_width, bboxes[j][1]/img_height, bboxes[j][2]/img_width - bboxes[j][0]/img_width, bboxes[j][3]/img_height - bboxes[j][1]/img_height, movingRate, handDist)
        # items += '{},{},{}/{}/{}/{},{},{}\n'.format(class_names[classes[j]], classes[j], bboxes[j][0]/img_width, bboxes[j][1]/img_height, bboxes[j][2]/img_width - bboxes[j][0]/img_width, bboxes[j][3]/img_height - bboxes[j][1]/img_height, movingRate, "far")        
    if(items == ''): items = ' '


    if(isDebug):
        moved_object_image = np.zeros((img_height, img_width), dtype=np.uint8)
        for j in range(len(masks)):
            mask = (masks[j]*255).astype(np.uint8)
            if(diff_from_env[j] >= 7.5 and diff_from_hand[j] <= 4.0): moved_object_image = cv.bitwise_or(moved_object_image, mask)
        cv.imwrite('result/moved_object_image.jpg', moved_object_image)

    moved_object_image = np.zeros((img_height, img_width), dtype=np.uint8)
    for j in range(len(masks)):
        mask = (masks[j]*255).astype(np.uint8)
        if(class_names[classes[j]] == "cup" or class_names[classes[j]] == "person"): moved_object_image = cv.bitwise_or(moved_object_image, mask)
    # cv.imwrite('data_0531/camera_image_{:0>4}_object.jpg'.format(cnt), moved_object_image)

    return items

def printItems(classes, bboxes):
    sorted_items = sorted([x for x in zip(classes, bboxes)], key=lambda x:x[1][0])
    items = ''
    # for j in range(len(classes)):
    #     items += '{},{},{}/{}/{}/{},{},{}\n'.format(class_names[classes[j]], classes[j], bboxes[j][0]/img_width, bboxes[j][1]/img_height, bboxes[j][2]/img_width - bboxes[j][0]/img_width, bboxes[j][3]/img_height - bboxes[j][1]/img_height, 0, "far")    
    for j in sorted_items:
        items += '{},{},{}/{}/{}/{},{},{}\n'.format(class_names[j[0]], j[0], j[1][0]/img_width, j[1][1]/img_height, j[1][2]/img_width - j[1][0]/img_width, j[1][3]/img_height - j[1][1]/img_height, 0, "far")    
    if(items == ''):
        items = ' ' 
    return items

if __name__ == '__main__':
    os.makedirs('masks', exist_ok=True)
    os.makedirs('result', exist_ok=True)
    os.makedirs('result/moved_objects', exist_ok=True)

    isDebug = False
    global img_width
    global img_height
    # img_width = 640
    # img_height = 384
    # img_width = 424
    # img_height = 256
    # img_width = 336
    # img_height = 200
    img_width = 256
    img_height = 152
    # img_width = 550
    # img_height = 330

    shape = (img_height, img_width, 3)
    img_size = np.prod(shape)
    mm_image_in = mmap_manager.mmapManager('./flow_img_in.dat', img_size, shape)
    mm_image_out = mmap_manager.mmapManager('./flow_img_out.dat', img_size, shape)
    mm_status = mmap_manager.mmapManager('./status.dat', 4)
    mm_status.WriteString('frst')

    trained_model='yolact/weights/yolact_plus_resnet50_54_800000.pth'
    net = gen_mask.prepare_net(trained_model)
    global class_names
    class_names = gen_mask.get_class_names()

    port = 4000
    serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversock.bind(('',port))
    serversock.listen(10)
    print('Waiting for connections...')
    clientsock, client_address = serversock.accept()
    print("Succeeded in Connection!")
    print(client_address)

    global cnt
    cnt = 0

    # images = glob.glob(os.path.join(sys.path[0], 'demo-frames', '*.png')) + \
    #         glob.glob(os.path.join(sys.path[0], 'demo-frames', '*.jpg'))
    # images = sorted(images)
    # for imfile in images:
    try:
        while True:
            cnt += 1
            print(cnt)

            # cam_image = cv.imread(imfile)
            buffer_size = int.from_bytes(clientsock.recv(8), 'little') # int64 / 8 = 8
            clientsock.send(bytes('received', 'utf-8'))
            data = b''
            while len(data) < buffer_size :
                try:
                    diffdata = clientsock.recv(buffer_size)
                except:
                    print("exception!")
                    diffdata = b''
                data += diffdata
            tmp = np.frombuffer(data, np.uint8, -1)
            try:
                img = cv.imdecode(tmp, cv.IMREAD_COLOR)
            except:
                clientsock.send(bytes('person 0 0.253125 0.559896 0.240625 0.4375 0.679568', 'utf-8'))
                continue
            cam_image = cv.resize(img, dsize=(640, 384), interpolation = cv.INTER_NEAREST)
            cam_image = cv.resize(cam_image, dsize=(img_width, img_height), interpolation = cv.INTER_NEAREST)
            if(isDebug): cv.imwrite('result/camera_image.jpg', cam_image)
            # cv.imwrite('data_0531/camera_image_{:0>4}.jpg'.format(cnt), cam_image)

            mm_image_in.WriteImage(cam_image)
            if cnt == 1:
                mm_status.WriteString('frst')
            else:
                mm_status.WriteString('sent')

            # generate mask
            masks, bboxes, classes = gen_mask.gen_mask(net, cam_image)
            while mm_status.ReadString() in ['sent', 'frst']:
                pass
            flow = mm_image_out.ReadImage()

            if cnt > 1:
                start_iou = time.perf_counter()
                items = calc_iou(flow, masks, bboxes, classes, isDebug)
                stop_iou = time.perf_counter()
                # print("Calculate IoU:", (stop_iou - start_iou) * 1000, "ms")
                # with open("runs/detect/exp/labels/out.txt", "r") as file:
                #     inferred_labels = file.read()
                clientsock.send(bytes(items, 'utf-8'))
                # print(items)
            else: 
                print('')
                clientsock.send(bytes('person 0 0.253125 0.559896 0.240625 0.4375 0.679568', 'utf-8'))
    except KeyboardInterrupt:
        mm_image_in.dispose()
        mm_image_out.dispose()
        mm_status.dispose()