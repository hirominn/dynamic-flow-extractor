import time
import mmap
import argparse
import cv2 as cv
import numpy as np

from util import mmap_manager
from RAFT import gen_flow

if __name__ == '__main__':
    shape = (384, 640, 3)
    img_size = np.prod(shape)
    mm_image_in = mmap_manager.mmapManager('./flow_img_in.dat', img_size)
    mm_image_out = mmap_manager.mmapManager('./flow_img_out.dat', img_size)
    mm_status = mmap_manager.mmapManager('./status.dat', 4)

    raft_args = argparse.Namespace(model='RAFT/models/raft-things.pth', path='RAFT/demo-frames', small=False, mixed_precision=True, alternate_corr=False)
    raft_model = gen_flow.load_model(raft_args)

    cnt = 0
    while True:
    # while cnt < 10:
        start = time.perf_counter()

        start_wait = time.perf_counter()

        while mm_status.ReadString() == 'rply':
            continue
        cnt += 1

        stop_wait = time.perf_counter()
        print("Wait:", (stop_wait - start_wait) * 1000, "ms")

        # image1 = gen_flow.load_image_from_file('demo-frames/out_0067.jpg')
        # image2 = gen_flow.load_image_from_file('demo-frames/out_0068.jpg')

        status = mm_status.ReadString()
        if status == 'frst':
            print('hi')
            image_now = gen_flow.load_image_from_cv(mm_image_in.ReadImage())
            mm_status.WriteString('rply')
            continue
        else:
            print('HI')
            image_pre = image_now
            start_load = time.perf_counter()
            image_now = gen_flow.load_image_from_cv(mm_image_in.ReadImage())
            stop_load = time.perf_counter()
            print("Loading Image:", (stop_load - start_load) * 1000, "ms")
            mm_image_out.WriteImage(gen_flow.demo(raft_args, raft_model, image_now, image_pre))

        # cv.imwrite("result.png", img)

        mm_status.WriteString('rply')

        stop = time.perf_counter()
        print("Flow Generation:", (stop - start) * 1000, "ms")
    mm_image_in.dispose()
    mm_image_out.dispose()
    mm_status.dispose()