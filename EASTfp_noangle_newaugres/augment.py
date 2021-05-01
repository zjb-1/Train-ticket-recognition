import numpy as np
import cv2 as cv
from numpy import random
from PIL import Image


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class ConvertFromFloat(object):
    def __call__(self, image, boxes=None, labels=None):
        image = np.clip(image, 0.0, 255.0)
        return image.astype(np.uint8), boxes, labels


class RandomBrightness(object):
    def __init__(self, delta = 30.0):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class RandomSampleCrop(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, image, boxes, labels=None):
        remain_w = image.shape[1] - self.length
        remain_h = image.shape[0] - self.length
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)

        ##crop_region = image.crop((start_w, start_h, start_w + self.length, start_h + self.length))
        crop_region = image[start_h:(start_h + self.length), start_w:(start_w + self.length)].copy()

        # # keep overlap with gt box IF center in sampled patch
        # centers = (boxes[:, [0, 1]] + boxes[:, [4, 5]]) / 2.0
        # ## box 的中心点  框内的
        # m1 = (start_w < centers[:, 0]) * (start_h < centers[:, 1])
        # m2 = ((start_w + self.length) > centers[:, 0]) * ((start_h + self.length) > centers[:, 1])
        #
        # mask = m1 * m2
        #
        # current_boxes = boxes[mask, :].copy()
        # current_labels = labels[mask]
        #
        # ##找出 完全在 crop 中的 boxes
        # m3 = (current_boxes[:, 0] >= start_w) * (current_boxes[:, 4] <= (start_w + self.length))
        # m4 = (current_boxes[:, 1] >= start_h) * (current_boxes[:, 5] <= (start_h + self.length))
        #
        # ## m3*m4为完全在crop的 boxes， 1 - m3*m4为被cross的boxes
        # mask_cross = (1 - m3 * m4).astype(bool)
        # current_labels[mask_cross] = 0
        #
        # current_boxes[:, [0, 6]] = np.maximum(current_boxes[:, [0, 6]], start_w)
        # current_boxes[:, [0, 6]] -= start_w
        # current_boxes[:, [1, 3]] = np.maximum(current_boxes[:, [1, 3]], start_h)
        # current_boxes[:, [1, 3]] -= start_h
        #
        # current_boxes[:, [2, 4]] = np.minimum(current_boxes[:, [2, 4]], start_w+self.length)
        # current_boxes[:, [2, 4]] -= start_w
        # current_boxes[:, [5, 7]] = np.minimum(current_boxes[:, [5, 7]], start_h+self.length)
        # current_boxes[:, [5, 7]] -= start_h
        #
        # return crop_region, current_boxes, current_labels

        ##只要 boxes与 crop_img有重叠 ，就保留
        m1 = (start_w < boxes[:, 4]) * (start_h < boxes[:, 5])
        m2 = (boxes[:, 0] < (start_w + self.length)) * (boxes[:, 1] < (start_h + self.length))
        mask = m1 * m2

        current_boxes = boxes[mask, :].copy()

        current_labels = labels[mask]

        # ## 法1：找出 完全在 crop 中的 boxes ，常数为余量
        # m3 = (current_boxes[:, 0] >= (start_w-10)) * (current_boxes[:, 4] <= (start_w + self.length+10))
        # m4 = (current_boxes[:, 1] >= (start_h-10)) * (current_boxes[:, 5] <= (start_h + self.length+10))
        # ### m3*m4为完全在crop的 boxes， 1 - m3*m4为被cross的boxes
        # mask_cross = (1 - m3 * m4).astype(bool)
        # current_labels[mask_cross] = 0

        ## 法2： 找到 中心点在 crop 中的 boxes
        centers = (current_boxes[:, [0, 1]] + current_boxes[:, [4, 5]]) / 2.0
        m3 = (start_w < centers[:, 0]) * (start_h < centers[:, 1])
        m4 = ((start_w + self.length) > centers[:, 0]) * ((start_h + self.length) > centers[:, 1])
        ### m3*m4为 中心点 在crop的 boxes， 1 - m3*m4 为需要ignore的boxes
        mask_cross = (1 - m3 * m4).astype(bool)
        current_labels[mask_cross] = 0


        current_boxes[:, [0, 6]] = np.maximum(current_boxes[:, [0, 6]], start_w)
        current_boxes[:, [0, 6]] -= start_w
        current_boxes[:, [1, 3]] = np.maximum(current_boxes[:, [1, 3]], start_h)
        current_boxes[:, [1, 3]] -= start_h

        current_boxes[:, [2, 4]] = np.minimum(current_boxes[:, [2, 4]], start_w+self.length)
        current_boxes[:, [2, 4]] -= start_w
        current_boxes[:, [5, 7]] = np.minimum(current_boxes[:, [5, 7]], start_h+self.length)
        current_boxes[:, [5, 7]] -= start_h

        return crop_region, current_boxes, current_labels


class RandomScale(object):
    '''
    first expand the image to 1~1.3 times
    then  resize the image to length
    Aim to get smaller words
    '''
    def __init__(self, length):
        self.length = length

    def __call__(self, image, boxes, labels=None):
        random_ratio = random.uniform(1, 1.5)
        h, w, channels = image.shape

        left = random.uniform(0, w * random_ratio - w)
        top = random.uniform(0, h * random_ratio - h)

        expand_image = np.zeros(
            (int(h * random_ratio), int(w * random_ratio), channels),
            dtype=image.dtype)
        expand_image[int(top):int(top + h),
                     int(left):int(left + w)] = image

        boxes = boxes.copy()
        boxes[:, [0, 2, 4, 6]] += int(left)
        boxes[:, [1, 3, 5, 7]] += int(top)

        res_image = cv.resize(expand_image, (self.length, self.length))
        ratio_h = self.length / expand_image.shape[0]
        ratio_w = self.length / expand_image.shape[1]

        boxes[:, [0, 2, 4, 6]] = boxes[:, [0, 2, 4, 6]] * ratio_w
        boxes[:, [1, 3, 5, 7]] = boxes[:, [1, 3, 5, 7]] * ratio_h

        return res_image, boxes, labels


def SSDAugmentation(img, boxes, labels, min_len, size):
    assert min_len >= size, 'min_len must be greater than or equal to crop_size'

    augment = Compose([
            ConvertFromInts(),        ## uint8 --> float32
            RandomBrightness(),
            RandomSampleCrop(size),   ##random crop to size * size
            RandomScale(size),        ##random scale, then resize to size * size
            ConvertFromFloat()        ## float32 --> uint8
        ])

    return augment(img, boxes, labels)


###############################################
def extract_vertices(lines):
    labels = []
    vertices = []
    for line in lines:
        vertices.append(list(map(int, line.rstrip('\n').lstrip('\ufeff').split(',')[:8])))
        label = 1
        labels.append(label)
    return np.array(vertices), np.array(labels)


if __name__ == '__main__':
    img_path = r'G:\Dataset\hcp_detection\train_images\train__89_iphone8p_12_113_1509_crop.jpg'
    gt_path = r'G:\Dataset\hcp_detection\train_gts\train__89_iphone8p_12_113_1509_crop.txt'
    with open(gt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    vertices, labels = extract_vertices(lines)
    img = cv.imread(img_path)

    a, b, c = SSDAugmentation(img, vertices, labels, 1024, 512)

    for i, box in enumerate(b):
        if c[i] == 1:
            cv.rectangle(a, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), (255, 0, 0))
        else:
            cv.rectangle(a, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), (0, 255, 0))

    cv.imshow('1', a)
    cv.waitKey()



