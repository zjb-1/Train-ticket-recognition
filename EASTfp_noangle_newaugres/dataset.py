from shapely.geometry import Polygon
import numpy as np
import cv2 as cv
from PIL import Image
import math
import os
import torch
import torchvision.transforms as transforms
from torch.utils import data
from augment import SSDAugmentation


def cal_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def move_points(vertices, index1, index2, r, coef):
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices


def shrink_poly(vertices, coef = 0.1):
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1, y1, x2, y2), cal_distance(x1, y1, x4, y4))
    r2 = min(cal_distance(x2, y2, x1, y1), cal_distance(x2, y2, x3, y3))
    r3 = min(cal_distance(x3, y3, x2, y2), cal_distance(x3, y3, x4, y4))
    r4 = min(cal_distance(x4, y4, x1, y1), cal_distance(x4, y4, x3, y3))
    r = [r1, r2, r3, r4]

    if cal_distance(x1,y1,x2,y2) + cal_distance(x3,y3,x4,y4) > \
        cal_distance(x1, y1, x4, y4) + cal_distance(x2, y2, x3, y3):
        offset = 0
    else:
        offset = 1

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v


def get_rotate_mat(theta):
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    v = vertices.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def get_boundary(vertices):
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


def cal_error(vertices):
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err


def find_min_rect_angle(vertices):
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta/180*math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1,x2,x3,x4) - min(x1,x2,x3,x4))* \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k:area_list[k])   ##从小到大排序的 index
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index]/180*math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index]/180*math.pi


def is_cross_text(start_loc, length, vertices):
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array([start_w, start_h, start_w + length, start_h,\
                  start_w + length, start_h + length, start_w, start_h + length]).reshape((4, 2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99:
            return True

    return False


def crop_img(img, vertices, labels, length, index):
    h, w = img.height, img.width
    if h >= w and w < length:
        img = img.resize((length, int(h * length/w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length/h), length), Image.BILINEAR)
    ratio_w = img.width / w
    ratio_h = img.height / h
    assert(ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h

    remain_w = img.width - length
    remain_h = img.height - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[labels==1, :])

    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)  #剪裁图片
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:, [0, 2, 4, 6]] -= start_w
    new_vertices[:, [1, 3, 5, 7]] -= start_h
    return region, new_vertices

    # try:
    #     h, w = img.height, img.width
    #     if h >= w and w < length:
    #         img = img.resize((length, int(h * length/w)), Image.BILINEAR)
    #     elif h < w and h < length:
    #         img = img.resize((int(w * length/h), length), Image.BILINEAR)
    #     ratio_w = img.width / w
    #     ratio_h = img.height / h
    #     assert(ratio_w >= 1 and ratio_h >= 1)
    #
    #     new_vertices = np.zeros(vertices.shape)
    #     if vertices.size > 0:
    #         new_vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w
    #         new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h
    #
    #     vertice_x = [np.min(new_vertices[:, [0, 2, 4, 6]]), np.max(new_vertices[:, [0, 2, 4, 6]])]
    #     vertice_y = [np.min(new_vertices[:, [1, 3, 5, 7]]), np.max(new_vertices[:, [1, 3, 5, 7]])]
    #     remain_w = [0, img.width - length]
    #     remain_h = [0, img.height - length]
    #     if vertice_x[1] > length:
    #         remain_w[0] = vertice_x[1] - length
    #     if vertice_x[0] < remain_w[1]:
    #         remain_w[1] = vertice_x[0]
    #     if vertice_y[1] > length:
    #         remain_h[0] = vertice_y[1] - length
    #     if vertice_y[0] < remain_h[1]:
    #         remain_h[1] = vertice_y[0]
    #
    #     start_w = int(np.random.rand() * (remain_w[1] - remain_w[0])) + remain_w[0]
    #     start_h = int(np.random.rand() * (remain_h[1] - remain_h[0])) + remain_h[0]
    #
    #     box = (start_w, start_h, start_w + length, start_h + length)
    #     region = img.crop(box)  #剪裁图片
    #     if new_vertices.size == 0:
    #         return region, new_vertices
    #
    #     new_vertices[:, [0, 2, 4, 6]] -= start_w
    #     new_vertices[:, [1, 3, 5, 7]] -= start_h
    # except IndexError:
    #     print("\n crop_img function index error!!\n , img is %d"%(index))
    # else:
    #     pass
    # return region, new_vertices


def img_resize(img, vertices, min_len):
    h, w, _ = img.shape

    min_bian = h if h <= w else w
    ## 短边 归一化到 min_len
    ratio = min_len / min_bian

    re_h = ((ratio * h) // 32 ) * 32
    re_h = int(re_h)

    re_w = ((ratio * w) // 32 ) * 32
    re_w = int(re_w)

    img = cv.resize(img, (re_w, re_h))

    ratio_w = img.shape[1] / w
    ratio_h = img.shape[0] / h

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h
    return img, new_vertices


def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)  ##进行数组拼接
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) +\
                    np.array([[anchor_x], [anchor_y]])

    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def adjust_height(img, vertices, ratio=0.2):
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.shape[0]
    new_h = int(np.around(old_h * ratio_h))  ##四舍五入
    img = cv.resize(img, (img.shape[1], new_h))


    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]]*(new_h/old_h)
    return img, new_vertices


###  里面用的是  PIL.Image的方法，  还没有改成 opencv的方法！！
def rotate_img(img, vertices, angle_range=10):
    center_x = (img.width - 1)/2
    center_y = (img.height - 1)/2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertices in enumerate(vertices):
        new_vertices[i, :] = rotate_vertices(vertices, -angle/180*math.pi, np.array([[center_x],[center_y]]))
    return img, new_vertices


def get_score_geo(img, vertices, labels, scale):
    score_map   = np.zeros((int(img.shape[0] * scale), int(img.shape[1] * scale), 1), np.float32)
    geo_map     = np.zeros((int(img.shape[0] * scale), int(img.shape[1] * scale), 4), np.float32)
    ignored_map = np.zeros((int(img.shape[0] * scale), int(img.shape[1] * scale), 1), np.float32)

    # index_w = np.arange(0, img.width, int(1 / scale))
    # index_h = np.arange(0, img.height, int(1 / scale))
    # index_x, index_y = np.meshgrid(index_w, index_h)
    ignored_polys = []
    polys = []

    for i, vertice in enumerate(vertices):
        if labels[i] == 0:
            ignored_polys.append(np.around(scale * vertice.reshape((4, 2))).astype(np.int32))
            continue

        poly = np.around(scale * shrink_poly(vertice).reshape((4, 2))).astype(np.int32)
        polys.append(poly)
        temp_mask = np.zeros(score_map.shape[:-1], np.float32)
        cv.fillPoly(temp_mask, [poly], 1)

        x_min, x_max, y_min, y_max = get_boundary(vertice)

        x = np.arange(0, img.shape[1], int(1 / scale))      ##   scale * width
        y = np.arange(0, img.shape[0], int(1 / scale))      ##   scale * height
        rotated_x, rotated_y = np.meshgrid(x, y)     ## rotated_x, rotated_y : shape --> [scale * height,scale * width]

        d1 = rotated_y - y_min
        d1[d1 < 0] = 0
        d2 = y_max - rotated_y
        d2[d2 < 0] = 0
        d3 = rotated_x - x_min
        d3[d3 < 0] = 0
        d4 = x_max - rotated_x
        d4[d4 < 0] = 0
        geo_map[:, :, 0] += d1 * temp_mask
        geo_map[:, :, 1] += d2 * temp_mask
        geo_map[:, :, 2] += d3 * temp_mask
        geo_map[:, :, 3] += d4 * temp_mask

    cv.fillPoly(ignored_map, ignored_polys, 1)
    cv.fillPoly(score_map, polys, 1)
    return torch.Tensor(score_map).permute(2,0,1), torch.Tensor(geo_map).permute(2,0,1), torch.Tensor(ignored_map).permute(2,0,1)


def extract_vertices(lines):
    labels = []
    vertices = []
    for line in lines:
        vertices.append(list(map(int, line.rstrip('\n').lstrip('\ufeff').split(',')[:8])))
        label = 1
        labels.append(label)
    return np.array(vertices), np.array(labels)


SampleNum = 30
class custom_dataset(data.Dataset):
    def __init__(self, img_path, gt_path, min_len, crop_size, scale=0.25):
        super(custom_dataset, self).__init__()
        self.img_files = [os.path.join(img_path, img_file) for img_file in sorted(os.listdir(img_path))]
        self.gt_files = [os.path.join(gt_path, gt_file) for gt_file in sorted(os.listdir(gt_path))]
        self.scale = scale
        self.min_len = min_len
        self.crop_size = crop_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        with open(self.gt_files[index], 'r', encoding='utf-8') as f:
            lines = f.readlines()
        while(len(lines) < 1):                 ##应对 空标签的问题
            index = int(SampleNum * np.random.rand())
            with open(self.gt_files[index], 'r', encoding='utf-8') as f:
                lines = f.readlines()
        vertices, labels = extract_vertices(lines)
        #img = Image.open(self.img_files[index])   ## RGB

        img = cv.imread(self.img_files[index])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        t1, t2 = os.path.split(self.img_files[index])

        # img = np.array(img)[:, :, :3]
        # img = Image.fromarray(img)
        # img, vertices = adjust_height(img, vertices)
        # img, vertices = rotate_img(img, vertices)
        # img, vertices = crop_img(img, vertices, labels, self.min_len, index)

        img, vertices = img_resize(img, vertices, self.min_len)
        img, vertices, labels = SSDAugmentation(img, vertices, labels, self.min_len, self.crop_size)
        transform = transforms.Compose([transforms.ToTensor(),  #transforms.ColorJitter(0.5, 0.5, 0.5, 0.25) #从光强、饱和度等方面进行数据增强
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        score_map, geo_map, ignored_map = get_score_geo(img, vertices, labels, self.scale)
        return transform(img), score_map, geo_map, ignored_map


# if __name__ == '__main__':
#     train_img_path = 'G:/Dataset/ICDAR_2015/train_img'
#     train_gt_path  = 'G:/Dataset/ICDAR_2015/train_gt'
#     trainset = custom_dataset(train_img_path, train_gt_path)
#     train_loader = data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4,drop_last=True)
#
#     for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
#         print("{} is ok!!".format(i))