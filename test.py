import os
import sys
import torch
from thop import profile
import numpy as np
import cv2
from tqdm import tqdm
# import matlab.engine as engine
from ultralytics import YOLO
from utils import *
from scipy.spatial.distance import cdist
import math
from scipy.optimize import leastsq
from openpyxl import load_workbook


def detect_harris_corners(image, block_size=5, ksize=3, k=0.04):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image)
    harris_corners = cv2.cornerHarris(gray_image, block_size, ksize, k)
    # harris_corners = cv2.dilate(harris_corners, None)  # Dilate to mark the corners
    # image[harris_corners > 0.9 * harris_corners.max()] = [0, 0, 255]
    # image[harris_corners >= harris_corners.max()] = [0, 0, 255]
    corners = np.unravel_index(np.argmax(harris_corners), harris_corners.shape)
    return np.array(corners)


def detect_shi_tomasi_corners(image, max_corners=1, quality_level=0.01, min_distance=10):
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Shi-Tomasi 角点检测
    corners = cv2.goodFeaturesToTrack(gray_image, max_corners, quality_level, min_distance)

    return corners.squeeze()


def inference_by_yolo(xywhn, img, cols=8, rows=11, pw=None):
    h, w = img.shape[:2]
    centers = xywhn[:, :2]
    whn = xywhn[:, 2:4]
    dp1, dp2 = get_diag_pts(centers)
    dp3, dp4 = find_farthest_points(centers, dp1, dp2)
    # dp1, dp2 = find_farthest_points(centers, dp3, dp4)
    # dp3, dp4 = find_farthest_points(centers, dp1, dp2)
    # diag_points = [dp1, dp3, dp2, dp4]
    diag_points = sort_points_by_sum(np.array([dp1, dp2, dp3, dp4]), centers)
    d = np.array([[0, 0], [(cols-1) * 30, 0], [(cols-1) * 30, (rows-1) * 30], [0, (rows-1) * 30]]) * 0.1

    # colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    # for point, c in zip(np.array(diag_points), colors):
    #     cv2.circle(img, np.round([point[0]*w, point[1]*h]).astype(np.intp), 1, c, -1)
    # image = cv2.resize(img, (img.shape[1], img.shape[0]))
    # cv2.imshow('Top Gradient Points', image)
    # cv2.waitKey(0)

    centers_sort, whn_sort = sort_points_by_perspective(centers, whn, np.array(diag_points), d)
    # if pw is not None:
    #     centers_sort = opt_by_homography(centers_sort, pw)

    corners, wh = [], []
    for p, whn in zip(centers_sort, whn_sort):
        p_ = np.array([p[0] * w, p[1] * h])
        wh_ = np.array([whn[0] * w, whn[1] * h])
        corners.append(p_)
        wh.append(wh_)
    corners = np.array(corners)
    wh = np.array(wh)
    if pw is not None:
        corners_opt = opt_by_homography(corners, pw)
    else:
        corners_opt = corners

    # last_p = None
    # x = 2
    # length = 5
    # img = cv2.resize(img, (img.shape[1] * x, img.shape[0] * x))
    # for point, point_org in zip(np.array(corners_opt), corners):
    #     point = np.round([point[0] * x, point[1] * x]).astype(np.intp)
    #     point_org = np.round([point_org[0] * x, point_org[1] * x]).astype(np.intp)
    #     # cv2.circle(img, point, 1, (0, 0, 255), 2)
    #     # if last_p is not None:
    #     #     cv2.line(img, last_p, point, (0, 255, 0), 2)
    #     # last_p = point
    #     cv2.circle(img, point_org, 12, (0, 0, 255), 2)
    #     cv2.line(img, (point[0] - length, point[1]), (point[0] + length, point[1]), (0, 255, 0), 2)
    #     cv2.line(img, (point[0], point[1] - length), (point[0], point[1] + length), (0, 255, 0), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # # i = len(os.listdir('ours_corners_img/'))
    # # cv2.imwrite('ours_corners_img/output'+str(i+201).zfill(3)+'.png', img)

    return corners_opt, wh


def inference_by_yolo2(xywhn, img, cols=8, rows=11, pw=None):
    h, w = img.shape[:2]
    centers = xywhn[:, :2]
    whn = xywhn[:, 2:4]
    dp1, dp2 = get_diag_pts(centers)
    dp3, dp4 = find_farthest_points(centers, dp1, dp2)
    # dp1, dp2 = find_farthest_points(centers, dp3, dp4)
    # dp3, dp4 = find_farthest_points(centers, dp1, dp2)
    # diag_points = [dp1, dp3, dp2, dp4]
    diag_points = sort_points_by_sum(np.array([dp1, dp2, dp3, dp4]), centers)
    d = np.array([[0, 0], [(cols-1) * 30, 0], [(cols-1) * 30, (rows-1) * 30], [0, (rows-1) * 30]]) * 0.1
    centers_sort, whn_sort = sort_points_by_perspective(centers, whn, np.array(diag_points), d)
    corners, wh = [], []
    for p, whn in zip(centers_sort, whn_sort):
        p_ = np.array([p[0] * w, p[1] * h])
        wh_ = np.array([whn[0] * w, whn[1] * h])
        corners.append(p_)
        wh.append(wh_)
    corners = np.array(corners)
    wh = np.array(wh)
    if pw is not None:
        corners_opt = opt_by_homography2(corners, pw, img)
    else:
        corners_opt = corners
    return corners_opt, wh


# def inference_by_yolo_subpixel(xywhn, img, cols=8, rows=11, pw=None):
#     h, w = img.shape[:2]
#     centers = xywhn[:, :2]
#     whn = xywhn[:, 2:4]
#     dp1, dp2 = get_diag_pts(centers)
#     dp3, dp4 = find_farthest_points(centers, dp1, dp2)
#     diag_points = sort_points_by_sum(np.array([dp1, dp2, dp3, dp4]), centers)
#     d = np.array([[0, 0], [(cols-1) * 30, 0], [(cols-1) * 30, (rows-1) * 30], [0, (rows-1) * 30]]) * 0.1
#
#     # colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
#     # for point, c in zip(np.array(diag_points), colors):
#     #     cv2.circle(img, np.round([point[0]*w, point[1]*h]).astype(np.intp), 1, c, -1)
#     # image = cv2.resize(img, (img.shape[1], img.shape[0]))
#     # cv2.imshow('Top Gradient Points', image)
#     # cv2.waitKey(0)
#
#     centers_sort, whn_sort = sort_points_by_perspective(centers, whn, np.array(diag_points), d)
#
#     corners, corners_opt, wh = [], [], []
#     for p, whn in zip(centers_sort, whn_sort):
#         p_ = np.array([p[0] * w, p[1] * h])
#         wh_ = np.array([whn[0] * w, whn[1] * h])
#         corners.append(p_)
#         wh.append(wh_)
#
#         center_x = np.round(w * p[0]).astype(int)
#         center_y = np.round(h * p[1]).astype(int)
#         width = np.round(w * whn[0]).astype(int)
#         height = np.round(h * whn[1]).astype(int)
#         x1 = int(center_x - width / 2)
#         y1 = int(center_y - height / 2)
#         x2 = int(center_x + width / 2)
#         y2 = int(center_y + height / 2)
#         roi = img[y1:y2, x1:x2]
#         c = corner_sub_pixel_refinement(roi, np.array([width // 2, height // 2]), min(width, height)) + np.array([x1, y1])
#
#         corners_opt.append(c)
#     corners_opt = np.array(corners_opt)
#     corners = np.array(corners)
#     wh = np.array(wh)
#     # if pw is not None:
#     #     corners_opt = opt_by_homography(corners, pw)
#     # else:
#     #     corners_opt = corners
#
#     last_p = None
#     x = 2
#     img = cv2.resize(img, (img.shape[1] * x, img.shape[0] * x))
#     for point, point_org in zip(np.array(corners_opt), corners):
#         point = np.round([point[0] * x, point[1] * x]).astype(np.intp)
#         point_org = np.round([point_org[0] * x, point_org[1] * x]).astype(np.intp)
#         cv2.circle(img, point, 1, (0, 0, 255), 2)
#         cv2.circle(img, point_org, 7, (255, 0, 0), 1)
#         if last_p is not None:
#             cv2.line(img, last_p, point, (0, 255, 0), 1)
#         last_p = point
#     cv2.imshow('img', img)
#     cv2.waitKey(0)
#
#     return corners_opt, wh


def inference_by_yolo_H(xywhn, img, cols=8, rows=11):
    h, w = img.shape[:2]
    centers = xywhn[:, :2]
    whn = xywhn[:, 2:4]
    dp1, dp2 = get_diag_pts(centers)
    dp3, dp4 = find_farthest_points(centers, dp1, dp2)
    diag_points = sort_points_by_sum(np.array([dp1, dp2, dp3, dp4]), centers)
    d = np.array([[0, 0], [(cols-1) * 30, 0], [(cols-1) * 30, (rows-1) * 30], [0, (rows-1) * 30]]) * 0.1

    # colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    # for point, c in zip(np.array(diag_points), colors):
    #     cv2.circle(img, np.round([point[0]*w, point[1]*h]).astype(np.intp), 1, c, -1)
    # image = cv2.resize(img, (img.shape[1], img.shape[0]))
    # cv2.imshow('Top Gradient Points', image)
    # cv2.waitKey(0)

    centers_sort, whn_sort = sort_points_by_perspective_H(centers, whn, np.array(diag_points), d)

    # last_p = None
    # x = 4
    # img = cv2.resize(img, (img.shape[1]*x, img.shape[0]*x))
    # for point, point_org in zip(np.array(centers_sort), centers):
    #     point = np.round([point[0] * w*x, point[1] * h*x]).astype(np.intp)
    #     point_org = np.round([point_org[0] * w*x, point_org[1] * h*x]).astype(np.intp)
    #     cv2.circle(img, point_org, 1, (255, 0, 0), 3)
    #     cv2.circle(img, point, 1, (0, 0, 255), 3)
    #     if last_p is not None:
    #         cv2.line(img, last_p, point, (0, 255, 0), 1)
    #     last_p = point
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    corners, wh = [], []
    for p, whn in zip(centers_sort, whn_sort):
        p_ = np.array([p[0] * w, p[1] * h])
        wh_ = np.array([whn[0] * w, whn[1] * h])
        corners.append(p_)
        wh.append(wh_)
    corners = np.array(corners)
    wh = np.array(wh)
    return corners


def inference_by_yolo_shi_tomasi(xywhn, img, cols=8, rows=11, pw=None):
    h, w = img.shape[:2]
    centers = xywhn[:, :2]
    whn = xywhn[:, 2:4]
    dp1, dp2 = get_diag_pts(centers)
    dp3, dp4 = find_farthest_points(centers, dp1, dp2)
    diag_points = sort_points_by_sum(np.array([dp1, dp2, dp3, dp4]), centers)
    d = np.array([[0, 0], [(cols - 1) * 30, 0], [(cols - 1) * 30, (rows - 1) * 30], [0, (rows - 1) * 30]]) * 0.1
    centers_sort, whn_sort = sort_points_by_perspective(centers, whn, np.array(diag_points), d)

    corners = []
    for p, wh in zip(centers_sort, whn_sort):
        center_x = w * p[0]
        center_y = h * p[1]
        width = w * wh[0]
        height = h * wh[1]

        # Calculate box coordinates
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        # Crop the region of interest
        roi = img[y1:y2, x1:x2]
        c = detect_shi_tomasi_corners(roi) + np.array([x1, y1])
        corners.append(c)
    corners = np.array(corners)
    if pw is not None:
        corners = opt_by_homography(corners, pw)
    return corners


def inference_by_yolo_harris(xywhn, img, cols=8, rows=11, pw=None):
    h, w = img.shape[:2]
    centers = xywhn[:, :2]
    whn = xywhn[:, 2:4]
    dp1, dp2 = get_diag_pts(centers)
    dp3, dp4 = find_farthest_points(centers, dp1, dp2)
    diag_points = sort_points_by_sum(np.array([dp1, dp2, dp3, dp4]), centers)
    d = np.array([[0, 0], [(cols - 1) * 30, 0], [(cols - 1) * 30, (rows - 1) * 30], [0, (rows - 1) * 30]]) * 0.1
    centers_sort, whn_sort = sort_points_by_perspective(centers, whn, np.array(diag_points), d)

    corners = []
    for p, wh in zip(centers_sort, whn_sort):
        center_x = w * p[0]
        center_y = h * p[1]
        width = w * wh[0]
        height = h * wh[1]

        # Calculate box coordinates
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        # Crop the region of interest
        roi = img[y1:y2, x1:x2]
        c = detect_harris_corners(roi, k=0.04) + np.array([x1, y1])
        corners.append(c)
    corners = np.array(corners)

    if pw is not None:
        corners = opt_by_homography(corners, pw)

    return corners


def main(model_path, images_path, save):
    labels_path = './dataset/labels/test/'
    print(images_path.split('/')[-2])

    model = YOLO(model_path)

    model.fuse()

    cols = 8
    rows = 11
    square_size = 30
    # save_dir = './save/real/error/'
    # excel_path = 'runs_0116_enhanced/0116_resutls0313.xlsx'
    # workbook = load_workbook(excel_path)
    # s, xl_para, xl_flops, xl_fps, xl_miss, xl_fa, xl_rmse, xl_mre, xl_max_error, xl_std_error, xl_failed, xl_FPS = get_excel_location2(model_path, images_path)
    # sheet = workbook.worksheets[s]
    _, params, _, flops = model.info()
    print('flops:', flops)
    # if save:
    #     sheet[xl_para] = params / 1e6
    #     sheet[xl_flops] = flops
    #     workbook.save(excel_path)

    single_world_points = generate_world_points(rows, cols, square_size, 1).squeeze()
    world_points, corners_ = [], []
    success, failed = 0, 0
    miss_, fa_ = 0, 0
    time_consuming = 0.
    count = -1
    if images_path.endswith('test/'):
        example_inputs = torch.rand((1, 3, 512, 640)).to('cuda')
        print('begin warmup...')
        for i in tqdm(range(200), desc='warmup....'):
            model(example_inputs, verbose=False)
        for file in os.listdir(images_path):
            if file.endswith('.jpg') or file.endswith('.JPG') or file.endswith('.png'):
                img = cv2.imread(os.path.join(images_path, file))
                flag, xywhn, time0 = pred(model, img, cols, rows, conf=0.5, iou=0.4)
                count += 1
                time_consuming += time0
        FPS = count / time_consuming
        print('FPS:', FPS)
        # if save:
        #     sheet[xl_FPS] = FPS
        #     workbook.save(excel_path)
        return
    for file in os.listdir(images_path):
        if file.endswith('.jpg') or file.endswith('.JPG') or file.endswith('.png'):
            # 进行预测
            img = cv2.imread(os.path.join(images_path, file))
            h, w = img.shape[:2]
            flag, xywhn, time0 = pred(model, img, cols, rows, conf=0.5, iou=0.4)
            if count == -1:
                count += 1
            else:
                count += 1
                time_consuming += time0

            labels = read_yolo_labels(os.path.join(labels_path, file.replace('.png', '.txt').replace('.jpg', '.txt')))
            labels, _ = inference_by_yolo(labels, img, cols, rows)
            corners_true = []
            xy = []
            for label in labels:
                c1 = np.array([label[0], label[1]])
                corners_true.append(c1)
            for l_ in xywhn:
                c2 = np.array([l_[0]*w, l_[1]*h])
                xy.append(c2)
            #
            points_img = []
            points_wld = []
            miss, fa = 0, 0
            for c_ture, pw in zip(corners_true, single_world_points):
                d = 5
                c_find = None
                for corners in xy:
                    d_curr = calculate_distance(c_ture, corners)
                    if d_curr < d:
                        d = d_curr
                        c_find = corners
                if d < 5:
                    points_img.append(c_find)
                    points_wld.append(pw)
                else:
                    miss += 1
            points_wld = np.array(points_wld, dtype=np.float32)
            points_img = np.array(points_img, dtype=np.float32)
            fa = len(xy) - len(points_img)
            if fa + miss > 0:
                failed += 1
                print(file + ':', 'miss ' + str(miss), 'fa ' + str(fa))
                fa_ += fa
                miss_ += miss
            else:
                success += 1

            corners = points_img

            world_points.append(points_wld)
            corners_.append(corners)

            # img = view_corners(img, corners_yolo)
            # cv2.imshow('file', img)
            # cv2.waitKey(0)

    fps = count / time_consuming
    # print('FPS:', FPS)
    print('Miss:', miss_, 'FA:', fa_)
    print('Success:', success, 'Failed:', failed)

    # world_points = generate_world_points(rows, cols, square_size, success)
    t1 = time.time()
    ret1, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(world_points, corners_, (640, 512), None, None, flags=0)
    t2 = time.time()
    print('Camera Calibration Time 1:', t2 - t1)
    t1 = time.time()
    rmse1, mre1, max_e1, std_e1 = cal_error(world_points, corners_, K1, D1, rvecs1, tvecs1)
    t2 = time.time()
    print('Error Calculation Time 1:', t2 - t1)

    print('ours:', rmse1, mre1, max_e1, std_e1)
    # print('ours:\n', K1, D1)

    return rmse1, mre1, max_e1, std_e1

    # if save:
    #     sheet[xl_fps] = fps
    #     sheet[xl_miss] = miss_
    #     sheet[xl_fa] = fa_
    #     sheet[xl_rmse] = rmse1
    #     sheet[xl_mre] = mre1
    #     sheet[xl_max_error] = max_e1
    #     sheet[xl_std_error] = std_e1
    #     sheet[xl_failed] = failed
    #     workbook.save(excel_path)


if __name__ == '__main__':
    model_paths = [
        # 'weight/stage_1/weights/best.pt',
        # 'weight/stage_2_trainingset/weights/best.pt',
        'weight/stage_2_testset/weights/best.pt',
    ]

    images_paths = [
        './dataset/images/test1/',
        './dataset/images/test2/',
        './dataset/images/test3/',
        './dataset/images/test4/',
        './dataset/images/test5/',
    ]
    for model_path in model_paths:
        rmse_, mre_, max_e_, std_e_ = [], [], [], []
        for images_path in images_paths:
            rmse, mre, max_e, std_e = main(model_path, images_path, False)
            rmse_.append(rmse)
            mre_.append(mre)
            max_e_.append(max_e)
            std_e_.append(std_e)
        rmse_ = np.array(rmse_)
        mre_ = np.array(mre_)
        max_e_ = np.array(max_e_)
        std_e_ = np.array(std_e_)
        print(model_path)
        print('Average indicator:', np.mean(rmse_), np.mean(mre_), np.mean(max_e_), np.mean(std_e_))



