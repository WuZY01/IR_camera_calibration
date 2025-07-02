import os
import time
import numpy as np
import math
import cv2
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 24
# plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.labelweight'] = 'normal'


def point_to_line_distance(point, line_point1, line_point2):
    """计算点到线段的距离"""
    line_vec = line_point2 - line_point1
    point_vec = point - line_point1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0:
        return np.linalg.norm(point_vec)
    projection = np.dot(point_vec, line_vec) / line_len_sq
    projection = np.clip(projection, 0, 1)
    projection_point = line_point1 + projection * line_vec
    return np.linalg.norm(point - projection_point)


def find_farthest_points(all_points, diagonal_point1, diagonal_point2):
    """在对角线两侧分别找到距离对角线最远的点"""
    # 计算对角线的单位法向量
    diagonal_vec = diagonal_point2 - diagonal_point1
    normal = np.array([-diagonal_vec[1], diagonal_vec[0]])
    normal = normal / np.linalg.norm(normal)

    # 找到对角线两侧的最远点
    farthest_point_side1 = None
    farthest_point_side2 = None
    max_distance_side1 = -float('inf')
    max_distance_side2 = -float('inf')

    for point in all_points:
        distance = point_to_line_distance(point, diagonal_point1, diagonal_point2)
        side_value = np.dot(point - diagonal_point1, normal)

        if side_value > 0:
            # 点在对角线的一个侧面
            if distance > max_distance_side1:
                max_distance_side1 = distance
                farthest_point_side1 = point
        else:
            # 点在对角线的另一个侧面
            if distance > max_distance_side2:
                max_distance_side2 = distance
                farthest_point_side2 = point

    return farthest_point_side1, farthest_point_side2


def pred(model, img, cols, rows, conf=0.3, iou=0.3, savepath=None):
    t0 = time.time()
    results = model.predict(source=img, verbose=False, conf=conf, iou=iou)  # 进行预测
    t1 = time.time()
    # labels = results[0].names  # 获取标签名称
    xywhn = results[0].boxes.xywhn.cpu().numpy()  # 获取检测结果(像素)
    # xy = xywhn[:, :2]
    # _, mask = remove_outliers(xy, threshold=3)
    # df = pd.DataFrame(xywhn, columns=['x', 'y', 'w', 'h'])
    # xywhn = df[mask].values
    # if len(xywhn) != cols*rows:
    #     # print(len(xywhn))
    #     return False, xywhn, t1-t0
    #
    # # 获取所有框的中心点坐标 (x_center, y_center)
    # centers = xywhn[:, :2]  # 提取每个框的中心点 (x, y)
    # threshold = 2/512
    # # 遍历每一对框，计算中心点之间的距离
    # for i in range(len(centers)):
    #     for j in range(i + 1, len(centers)):  # 避免重复比较
    #         distance = calculate_distance(centers[i], centers[j])
    #         if distance < threshold:
    #             print(f"框 {i} 和框 {j} 的中心点距离为 {distance:.4f} 小于阈值 {threshold}.")
    #             return False, xywhn, t1-t0  # 一旦找到距离小于阈值的目标框，返回 True
    #
    # 如果重投影误差过大就开启这段代码
    x = 3
    height, width = img.shape[:2]
    img = cv2.resize(img, (width*x, height*x))
    # cv2.imwrite('E:/Calibration/1205/code/results/org.jpg', img)
    H, W, _ = img.shape
    for label in xywhn:
        # 解析每一行，提取xywhn格式的信息
        x_center, y_center, width, height = label

        # 将归一化的坐标转换为像素坐标
        x1 = int(np.round((x_center - width / 2) * W))
        y1 = int(np.round((y_center - height / 2) * H))
        x2 = int(np.round((x_center + width / 2) * W))
        y2 = int(np.round((y_center + height / 2) * H))

        # 绘制矩形框
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框，线宽2
        cv2.circle(img, (int(np.round(x_center * W)), int(np.round(y_center * H))), 2, (0, 0, 255), 2)

        # 显示图像
    # cv2.imwrite('E:/Calibration/1205/code/results/pred.jpg', img)
    if savepath is not None:
        cv2.imwrite(savepath, img)
    # cv2.imshow("Image with Bounding Boxes", img)
    # key = cv2.waitKey(0)
    # if key == ord('f') or key == ord('F'):  # 按下F键
    #     return False, xywhn, t1-t0
    #
    # return True, xywhn, t1-t0
    # # xy = xywhn[:, :2]
    # # wh = xywhn[:, 2:4]
    # # return xy, wh
    return True, xywhn, t1 - t0


def remove_outliers(coordinates, threshold=2):
    """
    Remove outliers from a list of 2D coordinates based on z-score.

    Parameters:
        coordinates (list of list): 2D list of coordinates, each being a pair [x, y].
        threshold (float): The z-score threshold for detecting outliers (default is 2).

    Returns:
        list: List of coordinates after removing outliers.
    """
    # 将输入数据转换为 pandas DataFrame
    df = pd.DataFrame(coordinates, columns=['x', 'y'])

    # 计算 z-score
    z_scores = np.abs(stats.zscore(df))

    # 创建一个布尔掩码，筛选出 z-score 小于阈值的点
    mask = (z_scores < threshold).all(axis=1)

    # 提取剔除离群点后的数据
    filtered_coordinates = df[mask].values

    return filtered_coordinates, mask


def judge_same_side(line, unordered_pts):
    """
    Function to determine if points are on the same side of a line or on the line itself.

    :param line: A dictionary with 'point1' and 'point2' keys, each containing a tuple of (x, y) coordinates.
    :param unordered_pts: A list of tuples, each representing the (x, y) coordinates of a point.
    :return: A boolean value, True if all points are on the same side of the line (including on the line), otherwise False.
    """
    # Extract points from the line
    x1, y1 = line['point1']
    x2, y2 = line['point2']

    # Extract x and y coordinates from the points list
    x = [pt[0] for pt in unordered_pts]
    y = [pt[1] for pt in unordered_pts]

    # Calculate the coefficients for the line equation: (y1 - y2) * x + (x2 - x1) * y - x2 * y1 + x1 * y2 = 0
    A = (y1 - y2) * np.array(x) + (x2 - x1) * np.array(y) - x2 * y1 + x1 * y2

    # Set a threshold to avoid errors in case points are exactly on the line
    threshold = 0.0

    # Check if all points are on the same side of the line (including on the line)
    flag = abs(sum(A + threshold >= 0)-sum(A + threshold <= 0))<=5

    return flag


def get_diag_pts(unordered_pts):
    """
    Function to find a pair of diagonal corners from a set of unordered points of a quadrilateral.

    :param unordered_pts: A list of tuples, each representing the (x, y) coordinates of a point.
    :return: Diag_corner1, Diag_corner2 - The coordinates of the diagonal corners.
    """
    # Sort points based on x and y coordinates
    x_sorted = sorted(enumerate(unordered_pts), key=lambda p: p[1][0])
    y_sorted = sorted(enumerate(unordered_pts), key=lambda p: p[1][1])

    # Get the points with the lowest and highest x and y coordinates
    left_corner = unordered_pts[x_sorted[0][0]]
    right_corner = unordered_pts[x_sorted[-1][0]]
    upper_corner = unordered_pts[y_sorted[0][0]]
    lower_corner = unordered_pts[y_sorted[-1][0]]
    # left_corner, right_corner, upper_corner, lower_corner = find_convex_quadrilaterals(unordered_pts)
    # return left_corner, right_corner, upper_corner, lower_corner

    # Create a list of corner points
    corners = [left_corner, right_corner, upper_corner, lower_corner]

    # Find the diagonal pair of corners
    for i in range(4):
        for j in range(4):
            corner1 = corners[i]
            corner2 = corners[j]
            if all(corner1 == corner2):
                continue
            line = {'point1': corner1, 'point2': corner2}
            if judge_same_side(line, unordered_pts):
                # If the points are not on the same side, they are diagonal corners
                return corner1, corner2
    corner1, corner2 = find_extreme_points(unordered_pts)
    return corner1, corner2


def find_extreme_points(points):
    """
    Finds the points with the minimum and maximum sum of coordinates in a set of points.

    :param points: A list of tuples, where each tuple contains (x, y) coordinates of a point.
    :return: A tuple containing two points (min_sum_point, max_sum_point).
    """
    # Initialize variables to store the points with extreme sums
    min_sum = float('inf')
    max_sum = float('-inf')
    min_sum_point = None
    max_sum_point = None

    # Iterate over the points to find the sums and the corresponding points
    for point in points:
        x, y = point
        current_sum = x + y

        # Update the points with minimum and maximum sums
        if current_sum < min_sum:
            min_sum = current_sum
            min_sum_point = point
        if current_sum > max_sum:
            max_sum = current_sum
            max_sum_point = point

    return min_sum_point, max_sum_point


def sort_points_by_sum(points, all_points):
    # 输入的前两个点连线是一条对角线，后两个点的连线是另一条对角线
    sorted_indices = [0, 2, 1, 3]
    result = points[sorted_indices]

    x1, y1 = result[0][0], result[0][1]
    x2, y2 = result[1][0], result[1][1]
    x = [pt[0] for pt in all_points]
    y = [pt[1] for pt in all_points]
    A = (y1 - y2) * np.array(x) + (x2 - x1) * np.array(y) - x2 * y1 + x1 * y2
    threshold = 0.01
    a = sum(A + threshold >= 0)
    b = sum(A + threshold <= 0)
    if a==0 or b==0:
        a = sum(A - threshold >= 0)
        b = sum(A - threshold <= 0)
    if not ((a>=80 and a<88) or (b>=80 and b<88)):
        order = [1, 4, 3, 2]
        final_order = [sorted_indices[i] for i in np.array(order) - 1]
        result = points[final_order]
    # print(a, b)

    return result


# 定义函数来计算点相对于 p1 的水平和垂直位置
def calculate_position(point, p1, p2, p4):
    x, y = point
    p1_x, p1_y = p1
    p2_x, p2_y = p2
    p4_x, p4_y = p4

    # 计算点的水平和垂直归一化位置
    horizontal_position = (x - p1_x) / (p2_x - p1_x) if p2_x != p1_x else 0
    vertical_position = (y - p1_y) / (p4_y - p1_y) if p4_y != p1_y else 0

    return vertical_position, horizontal_position


def sort_points_z_pattern(points, p1, p2, p3, p4):
    # 确保 p1, p2, p3, p4 是逆时针顺序
    # assert np.all(np.cross(p2 - p1, p3 - p1) > 0), "顶点必须是逆时针顺序"

    # 获取所有点的排序位置
    positions = np.array([calculate_position(point, p1, p2, p4) for point in points])

    # 按照垂直位置然后水平位置排序
    sorted_indices = np.lexsort((positions[:, 1], positions[:, 0]))
    sorted_points = points[sorted_indices]

    return sorted_points


def sort_points_by_perspective(points, wh, src_pts, dst_pts):
    """
    对点集进行透视变换并排序。

    参数:
    points (numpy.ndarray): 输入的点集，形状为 (N, 2)，N 是点的数量。
    src_pts (numpy.ndarray): 四个源点的坐标，形状为 (4, 2)。
    dst_pts (numpy.ndarray): 四个目标点的坐标，形状为 (4, 2)。

    返回:
    numpy.ndarray: 排序后的点集，形状为 (N, 2)。
    """
    # 确保点集和四个顶点都是 numpy 数组
    points = np.array(points, dtype='float32')
    src_pts = np.array(src_pts, dtype='float32')
    dst_pts = np.array(dst_pts, dtype='float32')

    # 计算透视变换矩阵
    matrix, _ = cv2.findHomography(src_pts, dst_pts)

    # 应用透视变换
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])
    transformed_points = np.dot(matrix, points_homogeneous.T).T
    transformed_points /= transformed_points[:, 2].reshape(-1, 1)  # 归一化
    transformed_points = transformed_points[:, :2]

    # 按从上到下，从左到右的原则排序
    transformed_points = np.round(transformed_points)
    sorted_points = points[np.lexsort((transformed_points[:, 0], transformed_points[:, 1]))]
    if wh is not None:
        sorted_wh = wh[np.lexsort((transformed_points[:, 0], transformed_points[:, 1]))]
    else:
        sorted_wh = None

    return sorted_points, sorted_wh


def sort_points_by_perspective_H(points, wh, src_pts, dst_pts):
    """
    对点集进行透视变换并排序。

    参数:
    points (numpy.ndarray): 输入的点集，形状为 (N, 2)，N 是点的数量。
    src_pts (numpy.ndarray): 四个源点的坐标，形状为 (4, 2)。
    dst_pts (numpy.ndarray): 四个目标点的坐标，形状为 (4, 2)。

    返回:
    numpy.ndarray: 排序后的点集，形状为 (N, 2)。
    """
    # 确保点集和四个顶点都是 numpy 数组
    points = np.array(points, dtype='float32')
    src_pts = np.array(src_pts, dtype='float32')
    dst_pts = np.array(dst_pts, dtype='float32')

    # 计算透视变换矩阵
    matrix, _ = cv2.findHomography(src_pts, dst_pts)

    # 应用透视变换
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])
    transformed_points = np.dot(matrix, points_homogeneous.T).T
    transformed_points /= transformed_points[:, 2].reshape(-1, 1)  # 归一化
    transformed_points = transformed_points[:, :2]

    # 按从上到下，从左到右的原则排序
    transformed_points = np.round(transformed_points)
    sorted_points = points[np.lexsort((transformed_points[:, 0], transformed_points[:, 1]))]
    sorted_wh = wh[np.lexsort((transformed_points[:, 0], transformed_points[:, 1]))]
    sorted_transformed_points = transformed_points[np.lexsort((transformed_points[:, 0], transformed_points[:, 1]))]

    # matrix2, _ = cv2.findHomography(sorted_transformed_points, sorted_points)
    #
    # ones = np.ones((sorted_transformed_points.shape[0], 1))
    # sorted_transformed_points_homogeneous = np.hstack([sorted_transformed_points, ones])
    # transformed_sorted_transformed_points = np.dot(matrix2, sorted_transformed_points_homogeneous.T).T
    # transformed_sorted_transformed_points /= transformed_sorted_transformed_points[:, 2].reshape(-1, 1)  # 归一化
    # transformed_sorted_transformed_points = transformed_sorted_transformed_points[:, :2]
    points_opt = opt_by_homography(sorted_points, sorted_transformed_points)

    return points_opt, sorted_wh


def opt_by_homography(points_image, points_world):
    points_world = points_world[:, :2]
    matrix, _ = cv2.findHomography(points_world, points_image)
    ones = np.ones((points_world.shape[0], 1))
    points_world_homogeneous = np.hstack([points_world, ones])
    transformed_points_world = np.dot(matrix, points_world_homogeneous.T).T
    transformed_points_world /= transformed_points_world[:, 2].reshape(-1, 1)  # 归一化
    transformed_points_world = transformed_points_world[:, :2]

    return transformed_points_world


def opt_by_homography2(points_image, points_world, image):
    points_world = points_world[:, :2]
    p1, p2 = [], []
    for a, b in zip(points_image, points_world):
        if all(b == [0, 0]) or all(b == [0, 300]) or all(b == [210, 0]) or all(b == [210, 300]):
            # box, ps = crop_square_center(image, a[0], a[1], 20)
            # a_sub = detect_grad_corners(box) + ps
            # p1.append(a_sub)
            p1.append(a)
            p2.append(b)
    p1 = np.array(p1)
    p2 = np.array(p2)
    matrix, _ = cv2.findHomography(p2, p1)
    # matrix, _ = cv2.findHomography(points_world, points_image)
    ones = np.ones((points_world.shape[0], 1))
    points_world_homogeneous = np.hstack([points_world, ones])
    transformed_points_world = np.dot(matrix, points_world_homogeneous.T).T
    transformed_points_world /= transformed_points_world[:, 2].reshape(-1, 1)  # 归一化
    transformed_points_world = transformed_points_world[:, :2]

    return transformed_points_world


def crop_square_center(image, center_x, center_y, side_length):
    """
    截取以指定像素为中心的正方形图像块。

    :param image: 原图像
    :param center_x: 正方形中心的 x 坐标
    :param center_y: 正方形中心的 y 坐标
    :param side_length: 正方形的边长
    :return: 截取的图像块
    """
    center_x = round(center_x)
    center_y = round(center_y)

    # 计算正方形区域的边界
    half_side = side_length // 2
    start_x = max(center_x - half_side, 0)
    end_x = min(center_x + half_side, image.shape[1])
    start_y = max(center_y - half_side, 0)
    end_y = min(center_y + half_side, image.shape[0])

    # 防止超出边界，调整区域
    if start_x == 0:
        end_x = min(side_length, image.shape[1])
    if start_y == 0:
        end_y = min(side_length, image.shape[0])

    # 截取正方形区域
    cropped_image = image[start_y:end_y, start_x:end_x]
    # cv2.imshow('image', cropped_image)
    # cv2.waitKey(0)

    return cropped_image, np.array([start_x, start_y])


def calculate_distance(p1, p2):
    """
    计算两点之间的欧几里得距离

    :param p1: 第一个点的坐标 (x1, y1)
    :param p2: 第二个点的坐标 (x2, y2)
    :return: 两点之间的距离
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def line_intersection(p1, p2, q1, q2):
    """
    计算两条线段 (p1p2 和 q1q2) 的交点

    :param p1: 第一个线段的起点
    :param p2: 第一个线段的终点
    :param q1: 第二个线段的起点
    :param q2: 第二个线段的终点
    :return: 交点的坐标 (x, y)
    """

    def cross_product(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    def subtract(p, q):
        return (p[0] - q[0], p[1] - q[1])

    def add(p, q):
        return (p[0] + q[0], p[1] + q[1])

    def scale(p, s):
        return (p[0] * s, p[1] * s)

    p1p2 = subtract(p2, p1)
    q1q2 = subtract(q2, q1)
    q1p1 = subtract(p1, q1)

    denom = cross_product(p1p2, q1q2)

    if denom == 0:
        raise ValueError("The lines are parallel and do not intersect")

    num1 = cross_product(q1p1, q1q2)
    num2 = cross_product(p1p2, q1p1)

    t1 = num1 / denom
    t2 = num2 / denom

    intersection = add(p1, scale(p1p2, t1))

    return intersection


def line_equation(p1, p2):
    """
    Calculate the equation of the line passing through points p1 and p2.
    The equation is returned in the form of a dictionary with keys 'a', 'b', 'c'.
    """
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = p1[0] * p2[1] - p2[0] * p1[1]
    return {'a': A, 'b': B, 'c': -C}


def calculate_intersection(p1, p2, q1, q2):
    """
    Calculate the intersection point of two lines given by points p1, p2 and q1, q2.
    If the lines are parallel, return None.
    """
    # Calculate the coefficients of the equations of the lines
    line1 = line_equation(p1, p2)
    line2 = line_equation(q1, q2)

    # Check if the lines are parallel (A1 * B2 == A2 * B1)
    if line1['a'] * line2['b'] == line2['a'] * line1['b']:
        return None  # Lines are parallel

    # Calculate the determinant to find the intersection point
    determinant = line1['a'] * line2['b'] - line1['b'] * line2['a']

    if determinant == 0:
        return None  # Lines are coincident or no solution exists

    # Calculate the x and y coordinates of the intersection point
    x = (line1['b'] * line2['c'] - line2['b'] * line1['c']) / determinant
    y = (line1['c'] * line2['a'] - line2['c'] * line1['a']) / determinant

    return np.array([-x, -y], dtype=np.float32)


def draw_corners(image, corners):
    if corners is not None:
        corners = np.round(corners).astype(np.intp)  # 将角点坐标转为整数
        if len(corners.shape) == 1:
            x, y = corners[0], corners[1]
            cv2.circle(image, (x, y), 1, [0, 0, 255], -1)  # 绘制角点，红色圆点
        else:
            for corner in corners:
                x, y = corner[0], corner[1]
                cv2.circle(image, (x, y), 1, [0, 0, 255], -1)  # 绘制角点，红色圆点
    return image



def draw_rectangle(image, center, width=25, height=25, color=(0, 255, 0), thickness=1):
    """
    在图像上绘制矩形框。

    参数:
    - image: 要绘制的图像（numpy 数组）。
    - center: 矩形框的中心坐标 (x, y)。
    - width: 矩形框的宽度。
    - height: 矩形框的高度。
    - color: 矩形框的颜色 (B, G, R)。
    - thickness: 矩形框的线条厚度。
    """
    center_x, center_y = center
    top_left_x = int(center_x - width / 2)
    top_left_y = int(center_y - height / 2)
    bottom_right_x = int(center_x + width / 2)
    bottom_right_y = int(center_y + height / 2)

    # 在图像上绘制矩形框
    cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)

    return image


def view_corners(image, corners, wh_list=None):
    last_p = None
    if wh_list is not None:
        for p, wh in zip(corners, wh_list):
            p = np.round(p).astype(int)
            if last_p is not None:
                cv2.line(image, p, last_p, (0, 255, 0), 1)
            last_p = p
            image = draw_corners(image, p)
            image = draw_rectangle(image, p, wh[0], wh[1], color=(0, 255, 255))
    else:
        for p in corners:
            p = np.round(p).astype(int)
            if last_p is not None:
                cv2.line(image, p, last_p, (0, 255, 0), 1)
            last_p = p
            image = draw_corners(image, p)
    return image


def generate_world_points(rows, cols, square_size, n):
    # Generate world points
    points_world = []
    for i in range(n):
        pw = []
        for row in range(rows):
            for col in range(cols):
                corner_w = np.array([col * square_size, row * square_size, 0], dtype=np.float32)
                pw.append(corner_w)
        points_world.append(np.array(pw))
    return np.array(points_world, dtype=np.float32)


# def corner_sub_pixel_refinement(image, initial_point, window_size=11, iterations=5, epsilon=0.001):
#     """
#     亚像素角点提取算法实现。
#
#     参数:
#     - image: 棋盘格角点局部的图像区域，应为灰度图像。
#     - initial_point: 角点的初始估计坐标，格式为(x, y)。
#     - window_size: 搜索窗口的大小。
#     - iterations: 迭代次数。
#     - epsilon: 停止准则的精度。
#
#     返回:
#     - refined_point: 亚像素精度的角点坐标。
#     """
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     x, y = initial_point
#     p = np.array([[x], [y]], dtype=np.float32)
#
#     # 计算窗口内点的坐标
#     window_x = max(0, x - window_size // 2)
#     window_y = max(0, y - window_size // 2)
#     window_points = image[window_y:window_y + window_size, window_x:window_x + window_size]
#
#     for _ in range(iterations):
#
#         # 计算窗口内点的灰度梯度
#         dx = np.gradient(window_points, axis=1)
#         dy = np.gradient(window_points, axis=0)
#
#         # 构建雅可比矩阵
#         J = np.zeros((2, 2))
#         r = np.zeros((2, 1))
#
#         for i in range(window_size):
#             for j in range(window_size):
#                 J[0, 0] += dx[i, j] * dy[i, j]
#                 J[0, 1] += -dx[i, j] * dx[i, j]
#                 J[1, 0] += -dy[i, j] * dy[i, j]
#                 J[1, 1] += dx[i, j] * dy[i, j]
#                 r[0] += dy[i, j] * (window_points[i, j] - y)
#                 r[1] += -dx[i, j] * (window_points[i, j] - x)
#
#         # 计算最小二乘解
#         delta_p, _, _, _ = np.linalg.lstsq(J, r, rcond=None)
#
#         # 更新角点坐标
#         p += delta_p
#
#         # 检查停止条件
#         if np.linalg.norm(delta_p) < epsilon:
#             break
#
#     return tuple(p.ravel())


def corner_sub_pixel_refinement(image, initial_point, window_size=11):
    """
    根据图像中的角点附近像素和灰度梯度进行亚像素角点提取。

    参数:
    - image: 棋盘格角点局部的图像区域，应为灰度图像。
    - initial_point: 角点的初始估计坐标，格式为(x, y)。
    - window_size: 搜索窗口的大小。

    返回:
    - refined_point: 亚像素精度的角点坐标。
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 终止条件，当迭代次数超过一定次数或残差足够小
    max_iters = 30
    epsilon = 1e-6
    lambda_ = 1e-5  # 正则化参数，防止逆矩阵计算时出现问题

    # 初始化参数
    q = np.array(initial_point, dtype=float)
    A = np.zeros((2, 2))
    b = np.zeros((2, 1))

    for _ in range(max_iters):
        # 清零矩阵和向量以供下一次迭代
        A.fill(0)
        b.fill(0)

        # 计算窗口内点的坐标和梯度
        for i in range(window_size):
            for j in range(window_size):
                dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
                dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

                # 计算雅可比矩阵和残差向量
                A[0, 0] += dx[i, j] * dy[i, j]
                A[0, 1] += -dx[i, j] * dx[i, j]
                A[1, 0] += -dy[i, j] * dy[i, j]
                A[1, 1] += dx[i, j] * dy[i, j]
                b[0] += dy[i, j] * (image[i, j] - q[1])
                b[1] += -dx[i, j] * (image[i, j] - q[0])

        # 添加正则化项
        A += np.eye(2) * lambda_

        # 计算最小二乘解
        delta_q, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # 更新角点坐标
        q += delta_q.squeeze()

        # 检查停止条件
        if np.linalg.norm(delta_q) < epsilon:
            break

    return q


def detect_subpixel_corners(image, num_points=4, min_distance=5):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算 X 方向的梯度
    grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=7)

    # 计算 Y 方向的梯度
    grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=7)

    # # 转换回 uint8 类型
    # abs_grad_x = cv2.convertScaleAbs(grad_x)
    # abs_grad_y = cv2.convertScaleAbs(grad_y)
    #
    # # 计算梯度的总和
    # gradient = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    #
    # gradient = cv2.resize(gradient, (gradient.shape[1] * 4, gradient.shape[0] * 4))
    # cv2.imshow('Gradient', gradient)
    # cv2.waitKey(0)

    # 计算梯度方向（弧度）
    # grad_angle = math.atan2(grad_y, grad_x)

    # 计算梯度的幅值
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 获取边界像素
    edges = np.zeros_like(gray_image, dtype=bool)
    x = 3
    edges[0 + x, :] = True
    edges[-1 - x, :] = True
    edges[:, 0 + x] = True
    edges[:, -1 - x] = True
    edges[0:0 + x, :] = False
    edges[0 - x:-1, :] = False
    edges[:, 0:0 + x] = False
    edges[:, 0 - x:-1] = False
    edges[-1, :] = False
    edges[:, -1] = False


    edge_points = np.column_stack(np.nonzero(edges))
    edge_gradients = grad_magnitude[edges]
    # cv2.imshow('grad', grad_magnitude)
    # cv2.waitKey(0)

    # 按梯度值排序，选择前 num_points 个最大值
    indices = np.argsort(edge_gradients)[::-1]
    sorted_edge_points = edge_points[indices]
    sorted_edge_gradients = edge_gradients[indices]

    # 选择前 num_points 个点，并确保它们之间的距离符合要求
    selected_points = []
    for i in range(len(sorted_edge_points)):
        if len(selected_points) == num_points:
            break

        point = sorted_edge_points[i]
        if len(selected_points) == 0:
            selected_points.append(point)
        else:
            distances = cdist([point], selected_points, metric='euclidean')
            if np.all(distances >= min_distance):
                selected_points.append(point)

    # 计算每对坐标之间的距离
    distances = []
    for i in range(len(selected_points)):
        for j in range(i + 1, len(selected_points)):
            dist = calculate_distance(selected_points[i], selected_points[j])
            distances.append(((selected_points[i], selected_points[j]), dist))

    # 按距离降序排序，取前两对
    distances.sort(key=lambda x: x[1], reverse=True)
    top_two_pairs = distances[:2]

    # 计算这两对线段的交点
    (p1, p2), _ = top_two_pairs[0]
    (q1, q2), _ = top_two_pairs[1]
    intersection = calculate_intersection(p1, p2, q1, q2)

    selected_points = [p1, p2, q1, q2]
    gradxy_points = [np.array([grad_x[p[0]][p[1]], grad_y[p[0]][p[1]]]) for p in selected_points]
    grad_vecs = [np.array([p, [p[0]+g[1], p[1]+g[0]]]) for p, g in zip(selected_points, gradxy_points)]
    # grad_points = [math.degrees(math.atan2(grad_x[p[0]][p[1]], grad_y[p[0]][p[1]])) for p in selected_points]
    # d = obj_func(intersection, grad_vecs)

    # print(obj_func(intersection, grad_vecs), sum(obj_func(intersection, grad_vecs)))
    p_opt, _ = leastsq(obj_func, intersection, args=(grad_vecs), maxfev=1000)
    # print(obj_func(p_opt, grad_vecs), sum(obj_func(p_opt, grad_vecs)))
    # print('\n')

    # show results
    # scale = 2
    # image = cv2.resize(image, (image.shape[:2][1]*scale, image.shape[:2][0]*scale))
    # grad_vecs = [v * scale for v in grad_vecs]
    # for vec in grad_vecs:
    #     p1 = vec[0]
    #     p2 = vec[1]
    #     v1 = p2 - p1
    #     v1 = v1 / np.linalg.norm(v1)
    #     p2 = p1 + v1*10
    #     p1 = np.round(p1).astype(int)
    #     p2 = np.round(p2).astype(int)
    #     cv2.arrowedLine(image, (p1[1], p1[0]), (p2[1], p2[0]), (0, 255, 0), 1, tipLength=0.5)
    # selected_points.append(intersection)
    # selected_points.append(p_opt)
    # selected_points = [np.round(p * scale).astype(int) for p in selected_points]
    # colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    # for point, c in zip(selected_points, colors):
    #     # print(point)
    #     cv2.circle(image, (point[1], point[0]), 1, c, -1)
    # image = cv2.resize(image, (image.shape[1]*4, image.shape[0]*4))
    # cv2.imshow('Top Gradient Points', image)
    # cv2.waitKey(0)

    return p_opt


def obj_func(p, vecs):
    dot_products = []
    for vec in vecs:
        p1 = vec[0]
        p2 = vec[1]
        v1 = p1-p
        v2 = p2-p1
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        # dot_products.append(np.dot(v1, v2))
        dot_products.append(np.abs(np.dot(v1, v2)))
    # return [np.std(np.array(dot_products)), np.std(np.array(dot_products))]
    return [sum(dot_products), sum(dot_products)]


def detect_lines(image, min_line_length=50):
    """Detect lines in an image using Hough Transform and filter out lines shorter than min_line_length."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, 10, minLineLength=min_line_length, maxLineGap=10)
    return lines


def calculate_intersection2(line1, line2, slope_threshold=0.1):
    """Calculate the intersection point of two lines if their slopes are sufficiently different,
    and ensure that the intersection point lies within the bounds of the line segments."""
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    def line_params(x1, y1, x2, y2):
        """Returns line parameters a, b, c for line equation ax + by + c = 0"""
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        return a, b, c

    a1, b1, c1 = line_params(x1, y1, x2, y2)
    a2, b2, c2 = line_params(x3, y3, x4, y4)

    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None  # Lines are parallel or nearly parallel

    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det

    slope1 = a1 / b1 if b1 != 0 else float('inf')
    slope2 = a2 / b2 if b2 != 0 else float('inf')

    if abs(slope1 - slope2) < slope_threshold:
        return None  # Slopes are too similar

    def is_point_on_segment(px, py, x1, y1, x2, y2):
        """Check if point (px, py) is on the line segment from (x1, y1) to (x2, y2)"""
        return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)

    if (is_point_on_segment(x, y, x1, y1, x2, y2) and
        is_point_on_segment(x, y, x3, y3, x4, y4)):
        return np.array([x, y])  # The intersection is within both segments
    else:
        return None  # The intersection is not within both segments


def remove_nearby_points(points, distance_threshold, image_size):
    """Remove points that are within a certain distance of each other and filter out points outside the image bounds."""
    filtered_points = []
    points = np.array(points)
    image_width, image_height = image_size

    # Filter out points outside the image boundaries
    points = points[
        (points[:, 0] >= 0) & (points[:, 0] < image_width) &
        (points[:, 1] >= 0) & (points[:, 1] < image_height)
    ]

    while len(points) > 0:
        point = points[0]
        distances = np.linalg.norm(points - point, axis=1)
        close_points = points[distances < distance_threshold]
        avg_point = np.mean(close_points, axis=0)
        filtered_points.append(tuple(avg_point.astype(int)))
        points = points[distances >= distance_threshold]

    return filtered_points


def detect_lines_intersection(image):
    scale = 1
    image = cv2.resize(image, (image.shape[:2][1]*scale, image.shape[:2][0]*scale))

    lines = detect_lines(image, min_line_length=5)
    # print(len(lines))
    if lines is None or len(lines) < 2:
        return None

    intersections = []
    num_lines = len(lines)
    for i in range(num_lines):
        for j in range(i + 1, num_lines):
            p = calculate_intersection2(lines[i], lines[j], slope_threshold=1)
            if p is not None:
                intersections.append(p)
    if len(intersections) == 0:
        return None
    image_size = (image.shape[1], image.shape[0])
    filtered_intersections = remove_nearby_points(intersections, 10, image_size)

    # # scale = 4
    # # img = cv2.resize(image, (image.shape[:2][1]*scale, image.shape[:2][0]*scale))
    # p1 = np.round(np.array(filtered_intersections[0])*scale).astype(int)
    # cv2.circle(image, (p1[1], p1[0]), 1, (0, 255, 0), 1)
    # cv2.imshow('img', image)
    # cv2.waitKey(0)

    return np.array(filtered_intersections[0])/4


def read_yolo_labels(label_file_path):
    """
    读取YOLO标签文件并解析内容。

    :param label_file_path: 标签文件的路径
    :return: 一个列表，每个元素是一个字典，包含物体的类别和边界框信息
    """
    objects = []
    with open(label_file_path, 'r') as file:
        for line in file.readlines():
            # 去除行尾的换行符并分割字符串
            values = line.strip().split()
            if len(values) == 5:
                object_class, x_center, y_center, width, height = values
                object_info = np.array([
                    float(x_center),
                    float(y_center),
                    float(width),
                    float(height)
                ])
                objects.append(object_info)
            else:
                print(f"Warning: Invalid label format in line: {line}")
    return np.array(objects)


def cal_error(points_world, points_image, K, D, rvecs, tvecs, dir=None, picname=None):
    error_list = []
    error_x = []
    error_y = []
    corners_idea = []
    for point_world, p_, r, t in zip(points_world, points_image, rvecs, tvecs):
        points_reprj, _ = cv2.projectPoints(point_world, r, t, K, D)
        points_reprj = points_reprj.squeeze()
        p_ = p_.squeeze()
        for p_r, p in zip(points_reprj, p_):
            error_x.append(p_r[0] - p[0])
            error_y.append(p_r[1] - p[1])
            error = cv2.norm(p_r - p, cv2.NORM_L2SQR)
            error_list.append(error)
        corners_idea.append(points_reprj)
    rmse = np.sqrt(np.mean(np.array(error_list)))
    mre = np.mean(np.sqrt(np.array(error_list)))
    max_error = np.sqrt(np.max(np.array(error_list)))
    std_error = np.std(np.sqrt(np.array(error_list)))

    if dir is not None:
        # 创建一个新的图形
        plt.figure(figsize=(7, 6.5), dpi=400)

        # 绘制二维点
        plt.scatter(error_x, error_y, color='blue', marker='+')

        # 获取数据范围
        x_min, x_max = min(error_x), max(error_x)
        y_min, y_max = min(error_y), max(error_y)

        # 确保 x 和 y 的范围相同
        range_min = min(x_min, y_min)
        range_max = max(x_max, y_max)
        plt.gca().set_xlim(range_min*1.05, range_max*1.05)
        plt.gca().set_ylim(range_min*1.05, range_max*1.05)

        # 添加标题和标签
        plt.title('Reprojection error')
        plt.xlabel('X Coordinate (pixel)')
        plt.ylabel('Y Coordinate (pixel)')
        plt.tick_params(axis='both', which='major', labelsize=18)

        # 显示图例
        # plt.legend()

        # 保存图形
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        os.makedirs(dir, exist_ok=True)
        plt.savefig(os.path.join(dir, picname))
        # plt.show()

        plt.close('all')

        print(picname, range_min, range_max)

    return rmse, mre, max_error, std_error


def get_excel_location(model_path, test_images_path):
    column = None
    if 'yolov3t' in model_path:
        column = 'C'
    elif 'yolov3-tiny' in model_path:
        column = 'D'
    elif 'yolov5n' in model_path:
        column = 'E'
    elif 'yolox_tiny' in model_path:
        column = 'F'
    elif 'yolov6n' in model_path:
        column = 'G'
    elif 'yolov8n' in model_path:
        column = 'H'
    elif 'yolov9t' in model_path:
        column = 'I'
    elif 'yolov10n' in model_path:
        column = 'J'
    elif 'yolo11n' in model_path:
        column = 'K'
    elif 'yolo-corner1' in model_path:
        column = 'L'
    elif 'yolo-corner2' in model_path:
        column = 'M'
    elif 'yolo-corner3' in model_path:
        column = 'N'
    elif 'yolo-corner4' in model_path:
        column = 'O'
    else:
        assert False

    row = None
    if 'test_1' in test_images_path:
        row = 6
    elif 'test_2' in test_images_path:
        row = 14
    elif 'test_3' in test_images_path:
        row = 22
    elif 'test_4' in test_images_path:
        row = 30
    elif 'test_5' in test_images_path:
        row = 38
    else:
        assert False

    xl_fps = column + str(row)
    xl_miss = column + str(row + 1)
    xl_fa = column + str(row + 2)
    xl_rmse = column + str(row + 3)
    xl_mre = column + str(row + 4)
    xl_max_error = column + str(row + 5)
    xl_std_error = column + str(row + 6)
    xl_failed = column + str(row + 7)

    return xl_fps, xl_miss, xl_fa, xl_rmse, xl_mre, xl_max_error, xl_std_error, xl_failed


def get_excel_location2(model_path, test_images_path):
    sheet = 0
    column = None
    row = None

    if 'yolo-corner1-a' in model_path:
        sheet = 1  # sheet 2
    elif 'yolo-corner2-a' in model_path:
        sheet = 2  # sheet 3
    elif 'yolo-corner1-se' in model_path:
        sheet = 3
    elif 'yolo-corner2-se' in model_path:
        sheet = 4
    elif ('yolo-xiaorong4-test-loss' in model_path) or ('yolo-xiaorong4-train-loss' in model_path):
        sheet = 6
    elif ('yolo-xiaorong5-test-loss' in model_path) or ('yolo-xiaorong5-train-loss' in model_path):
        sheet = 7

    if sheet == 0:
        if 'yolov3t' in model_path:
            column = 'C'
        elif 'yolov3-tiny' in model_path:
            column = 'D'
        elif 'yolov5n' in model_path:
            column = 'E'
        elif 'yolox_tiny' in model_path:
            column = 'F'
        elif 'yolov6n' in model_path:
            column = 'G'
        elif 'yolov7-tiny' in model_path:
            column = 'H'
        elif 'yolov8n' in model_path:
            column = 'I'
        elif 'yolov9t' in model_path:
            column = 'J'
        elif 'yolov10n' in model_path:
            column = 'K'
        elif 'yolo11n' in model_path:
            column = 'L'
        elif 'yolov12n' in model_path:
            column = 'M'
        elif 'yolo-xiaorong1' in model_path:
            column = 'P'
        elif 'yolo-xiaorong2' in model_path:
            column = 'Q'
        elif 'yolo-xiaorong3' in model_path:
            column = 'R'
        elif 'yolo-xiaorong4' in model_path:
            column = 'S'
        elif 'yolo-xiaorong5' in model_path:
            column = 'T'
        elif 'yolo-corner1' in model_path:
            column = 'U'
        elif 'yolo-corner2' in model_path:
            column = 'V'
        else:
            assert False
    elif sheet == 1 or sheet == 2:
        if 'a7' in model_path:
            column = 'D'
        elif 'a9' in model_path:
            column = 'E'
        elif 'a11' in model_path:
            column = 'F'
        elif 'a13' in model_path:
            column = 'G'
        elif 'a15' in model_path:
            column = 'H'
        elif 'a17' in model_path:
            column = 'I'
        elif 'a19' in model_path:
            column = 'J'
        else:
            assert False
    elif sheet == 3 or sheet == 4:
        if 'se2' in model_path:
            column = 'D'
        elif 'se4' in model_path:
            column = 'E'
        elif 'se8' in model_path:
            column = 'F'
        elif 'se16' in model_path:
            column = 'G'
        elif 'se32' in model_path:
            column = 'H'
        else:
            assert False
    elif sheet == 5:
        if 'train-loss-org' in model_path:
            column = 'D'
        elif 'train-loss2-ciou' in model_path:
            column = 'E'
        elif 'train-loss3-ciou' in model_path:
            column = 'F'
        elif 'train-loss2-corneriou' in model_path:
            column = 'G'
        elif 'train-loss3-corneriou' in model_path:
            column = 'H'
        elif 'test-loss-org' in model_path:
            column = 'I'
        elif 'test-loss2-ciou' in model_path:
            column = 'J'
        elif 'test-loss3-ciou' in model_path:
            column = 'K'
        elif 'test-loss2-corneriou' in model_path:
            column = 'L'
        elif 'test-loss3-corneriou' in model_path:
            column = 'M'
        else:
            assert False
    elif sheet == 6 or sheet == 7:
        if 'train-loss-org' in model_path:
            column = 'I'
        elif ('train-loss2-ciou' in model_path) and ('50-50' in model_path):
            column = 'J'
        elif ('train-loss2-ciou' in model_path) and ('100-100' in model_path):
            column = 'K'
        elif 'test-loss-org' in model_path:
            column = 'L'
        elif ('test-loss2-ciou' in model_path) and ('50-50' in model_path):
            column = 'M'
        elif ('test-loss2-ciou' in model_path) and ('100-100' in model_path):
            column = 'N'
        else:
            assert False

    if 'test1' in test_images_path:
        row = 6
    elif 'test2' in test_images_path:
        row = 14
    elif 'test3' in test_images_path:
        row = 22
    elif 'test4' in test_images_path:
        row = 30
    elif 'test5' in test_images_path:
        row = 38
    elif 'test/' in test_images_path:
        row = 110
    else:
        assert False

    xl_para = column + str(3)
    xl_flops = column + str(4)
    xl_fps = column + str(row)
    xl_miss = column + str(row + 1)
    xl_fa = column + str(row + 2)
    xl_rmse = column + str(row + 3)
    xl_mre = column + str(row + 4)
    xl_max_error = column + str(row + 5)
    xl_std_error = column + str(row + 6)
    xl_failed = column + str(row + 7)
    xl_FPS = column + str(50)

    return sheet, xl_para, xl_flops, xl_fps, xl_miss, xl_fa, xl_rmse, xl_mre, xl_max_error, xl_std_error, xl_failed, xl_FPS

