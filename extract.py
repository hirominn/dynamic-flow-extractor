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

from PIL import Image
from PIL import ImageDraw

def calc_iou(image, flow, masks, bboxes, classes, pre_masks, pre_bboxes, pre_classes, isDebug):
    items = ''

    if(isDebug): cv.imwrite('result/flow.png', flow)
    env_flow = flow

    if len(masks) == 0:
        return items

    # end if the image has no hand
    if next((f for f in classes if f == 0), None) == None:
        print('no hands')
        return printItems(classes, bboxes)

    if sum(1 if i == 0 else 0 for i in classes) > 1:
        print('multi hands')
        return items

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

    isMoving = False
    movingItems = []
    for j in range(len(masks)):
        if(diff_from_env[j] >= 4.0 and diff_from_hand[j] <= 3.0 and not isHand(classes[j])): 
            print('{}, hand_dist:{}'.format(class_names[classes[j]],lenFromHand(bboxes[j], hand_box)))
            print('{}, x[0]:{},x[1]:{},diff_env:{:0>4},diff_hand:{:0>4}\n'.format(class_names[classes[j]],obj_flows[j][0] * 180 * 2, obj_flows[j][1] * 255, diff_from_env[j], diff_from_hand[j]))
            movingItems.append(j)
            isMoving = True
    if(isMoving):
        for j in movingItems:
            if(lenFromHand(bboxes[j], hand_box) < 0.20):
                handDist = "near"
            else:
                handDist = "far"
            movingRate = 1
            min_dist = 2
            cand = -1
            for i in range(len(pre_bboxes)):
                if classes[j] != pre_classes[i]:
                    continue
                pre_x, pre_y, now_x, now_y = pre_bboxes[i][0]/img_width, pre_bboxes[i][1]/img_height, bboxes[j][0]/img_width, bboxes[j][1]/img_height
                now_dist = math.sqrt((now_x - pre_x)**2 + (now_y - pre_y)**2)
                if min_dist > now_dist:
                    min_dist = now_dist
                    cand = i
            if cand == -1:
            items += '{},{},{}/{}/{}/{},{},{}\n'.format(class_names[classes[j]], classes[j], bboxes[j][0]/img_width, bboxes[j][1]/img_height, bboxes[j][2]/img_width - bboxes[j][0]/img_width, bboxes[j][3]/img_height - bboxes[j][1]/img_height, movingRate, handDist)
    else:
                items += '{},{},{}/{}/{}/{},{},{}\n'.format(class_names[pre_classes[cand]], pre_classes[cand], pre_bboxes[cand][0]/img_width, pre_bboxes[cand][1]/img_height, pre_bboxes[cand][2]/img_width - pre_bboxes[cand][0]/img_width, pre_bboxes[cand][3]/img_height - pre_bboxes[cand][1]/img_height, movingRate, handDist)
                print('{},{},{}/{}/{}/{},{},{}\n'.format(class_names[pre_classes[cand]], pre_classes[cand], pre_bboxes[cand][0]/img_width, pre_bboxes[cand][1]/img_height, pre_bboxes[cand][2]/img_width - pre_bboxes[cand][0]/img_width, pre_bboxes[cand][3]/img_height - pre_bboxes[cand][1]/img_height, movingRate, handDist))
    else:
        no_near = True
        no_near_len = 10
        for j in range(len(masks)):
            if(lenFromHand(bboxes[j], hand_box) < 0.20):
                handDist = "near"
                if(not isHand(classes[j])): no_near = False
            else:
                handDist = "far"
                if(not isHand(classes[j]) and no_near_len > lenFromHand(bboxes[j], hand_box)):
                    no_near_len = lenFromHand(bboxes[j], hand_box)
            movingRate = 0
            items += '{},{},{}/{}/{}/{},{},{}\n'.format(class_names[classes[j]], classes[j], bboxes[j][0]/img_width, bboxes[j][1]/img_height, bboxes[j][2]/img_width - bboxes[j][0]/img_width, bboxes[j][3]/img_height - bboxes[j][1]/img_height, movingRate, handDist)
        # Draw Dot
        # if(no_near): 
        #     cv.imwrite('data_0626/no_near_{:0>4}_camera_image.jpg'.format(cnt), image)            
        #     cv.imwrite('data_0626/no_near_{:0>4}_env_flow.jpg'.format(cnt), env_flow)           
        #     print(no_near_len)
        #     img = Image.open('data_0626/no_near_{:0>4}_camera_image.jpg'.format(cnt))
        #     d = ImageDraw.Draw(img)
        #     for i in range(len(classes)):
        #         if(classes[i] == 41):
        #             d.ellipse([((bboxes[i][0]+bboxes[i][2])/2-3, (bboxes[i][1]+bboxes[i][3])/2-3), ((bboxes[i][0]+bboxes[i][2])/2+3, (bboxes[i][1]+bboxes[i][3])/2+3)], fill='red', outline='lime', width=1)
        #     d.ellipse([(hand_box[0]-3, hand_box[1]-3), (hand_box[0]+3, hand_box[1]+3)], fill='blue', outline='lime', width=1)
        #     img.save('data_0626/no_near_{:0>4}_camera_image_dot.jpg'.format(cnt))

    if(isDebug):
        moved_object_image = np.zeros((img_height, img_width), dtype=np.uint8)
        for j in range(len(masks)):
            mask = (masks[j]*255).astype(np.uint8)
            if(diff_from_env[j] >= 7.5 and diff_from_hand[j] <= 4.0): moved_object_image = cv.bitwise_or(moved_object_image, mask)
        cv.imwrite('result/moved_object_image.jpg', moved_object_image)

    # moved_object_image = np.zeros((img_height, img_width), dtype=np.uint8)
    # for j in range(len(masks)):
    #     mask = (masks[j]*255).astype(np.uint8)
    #     if(class_names[classes[j]] == "cup" or class_names[classes[j]] == "person"): moved_object_image = cv.bitwise_or(moved_object_image, mask)
    # cv.imwrite('data_0531/camera_image_{:0>4}_object.jpg'.format(cnt), moved_object_image)

    return items

def isHand(classId):
    if(classId == 0): return True
    else: return False

def lenFromHand(itemBox, handBox):
    return np.sqrt(pow((itemBox[0]/img_width + itemBox[2]/img_width)/2 - handBox[0]/img_width, 2) + pow(itemBox[1]/img_height - handBox[1]/img_height, 2))

def printItems(classes, bboxes):
    sorted_items = sorted([x for x in zip(classes, bboxes)], key=lambda x:x[1][0])
    items = ''
    # for j in range(len(classes)):
    #     items += '{},{},{}/{}/{}/{},{},{}\n'.format(class_names[classes[j]], classes[j], bboxes[j][0]/img_width, bboxes[j][1]/img_height, bboxes[j][2]/img_width - bboxes[j][0]/img_width, bboxes[j][3]/img_height - bboxes[j][1]/img_height, 0, "far")    
    for j in sorted_items:
        items += '{},{},{}/{}/{}/{},{},{}\n'.format(class_names[j[0]], j[0], j[1][0]/img_width, j[1][1]/img_height, j[1][2]/img_width - j[1][0]/img_width, j[1][3]/img_height - j[1][1]/img_height, 0, "far")    
    return items
            
def mask_class(masks, bboxes, classes):
    allow_classes = [0, 41]
    new_masks, new_bboxes, new_classes = [], [], []
    for i in range(len(masks)):
        if classes[i] in allow_classes:
            new_masks.append(masks[i])
            new_bboxes.append(bboxes[i])
            new_classes.append(classes[i])
    return new_masks, new_bboxes, new_classes

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
            try:
            while len(data) < buffer_size :
                    diffdata = clientsock.recv(buffer_size)
                data += diffdata
            tmp = np.frombuffer(data, np.uint8, -1)
            except:
                print("exception!")
                continue
            try:
                img = cv.imdecode(tmp, cv.IMREAD_COLOR)
                cam_image = cv.resize(img, dsize=(640, 384), interpolation = cv.INTER_NEAREST)
                cam_image = cv.resize(cam_image, dsize=(img_width, img_height), interpolation = cv.INTER_NEAREST)
            except:
                clientsock.send(bytes('person 0 0.253125 0.559896 0.240625 0.4375 0.679568', 'utf-8'))
                continue
            cam_image = cv.resize(img, dsize=(640, 384), interpolation = cv.INTER_NEAREST)
            cam_image = cv.resize(cam_image, dsize=(img_width, img_height), interpolation = cv.INTER_NEAREST)
            if(isDebug): cv.imwrite('result/camera_image.jpg', cam_image)
            # cv.imwrite('data_0531/camera_image_{:0>4}.jpg'.format(cnt), cam_image)

            # generate mask
            masks, bboxes, classes = gen_mask.gen_mask(net, cam_image)
            masks, bboxes, classes = mask_class(masks, bboxes, classes)
            mm_image_in.WriteImage(cam_image)
            if cnt == 1:
                mm_status.WriteString('frst')
            else:
                mm_status.WriteString('sent')

            masks, bboxes, classes = gen_mask.gen_mask(net, cam_image)
            masks, bboxes, classes = mask_class(masks, bboxes, classes)

            while mm_status.ReadString() in ['sent', 'frst']:
                pass
            flow = mm_image_out.ReadImage()

            if cnt > 1:
                items = calc_iou(cam_image, flow, masks, bboxes, classes, pre_masks, pre_bboxes, pre_classes, isDebug)
                if(items == ''): items = ' '
                clientsock.send(bytes(items, 'utf-8'))
            else: 
                print('')
                clientsock.send(bytes('person 0 0.253125 0.559896 0.240625 0.4375 0.679568', 'utf-8'))
            pre_masks, pre_bboxes, pre_classes = masks, bboxes, classes
    except KeyboardInterrupt:
        mm_image_in.dispose()
        mm_image_out.dispose()
        mm_status.dispose()