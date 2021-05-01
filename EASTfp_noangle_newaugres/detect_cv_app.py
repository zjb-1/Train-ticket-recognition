import os
import re
import time
import torch
import numpy as np
import cv2 as cv
from cv2 import dnn
from torchvision import transforms
from PIL import Image, ImageDraw

from model import EAST
from model_MobileNetV2 import EAST_MobileV2
from nms import nms_locality
import argparse


parser = argparse.ArgumentParser(
    description='EAST Detector With Pytorch')
parser.add_argument('--min_len', default=800, type=int,
                    help='resize the smallest edge of the image to min_len')
args = parser.parse_args()


def resize_img(img):
    h, w, _ = img.shape

    min_bian = h if h <= w else w
    ## 短边 归一化到 min_len
    ratio = args.min_len / min_bian

    re_h = ((ratio * h) // 32 ) * 32
    re_h = int(re_h)

    re_w = ((ratio * w) // 32 ) * 32
    re_w = int(re_w)

    img = cv.resize(img, (re_w, re_h))

    ratio_w = img.shape[1] / w
    ratio_h = img.shape[0] / h

    return img, ratio_h, ratio_w


def xyxy2xywh(box_xyxy):
    box_xywh = np.zeros((box_xyxy.shape[0], 4), dtype=box_xyxy.dtype)
    box_xywh[:, :2] = box_xyxy[:, :2]
    box_xywh[:, 2:] = box_xyxy[:, [4, 5]] - box_xyxy[:, [0, 1]]

    return box_xywh


def xywh2xyxy(box_xywh):
    box_xyxy = np.zeros((box_xywh.shape[0], 8), dtype=box_xywh.dtype)
    box_xyxy[:, [0, 2, 4, 6]] = box_xywh[:, 0].reshape(box_xywh.shape[0], 1)
    box_xyxy[:, [1, 3, 5, 7]] = box_xywh[:, 1].reshape(box_xywh.shape[0], 1)
    box_xyxy[:, [2, 4]] += box_xywh[:, 2].reshape(box_xywh.shape[0], 1)
    box_xyxy[:, [5, 7]] += box_xywh[:, 3].reshape(box_xywh.shape[0], 1)

    return box_xyxy


def load_pil(img):
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return t(img).unsqueeze(0)


def is_valid_poly(res, score_shape, scale):
    cnt = 0
    for i in range(res.shape[1]):
        if cnt > 1:
            return False
        if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
                res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
            cnt += 1
    return True
    #return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    polys = []
    index = []
    valid_pos *= scale
    d = valid_geo[:4, :]

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]

        res = np.zeros((2, 4))
        res[0, :] = np.array([[x_min, x_max, x_max, x_min]])
        res[1, :] = np.array([[y_min, y_min, y_max, y_max]])

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1], res[0, 2], res[1, 2],res[0, 3], res[1, 3]])
    return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2, cv_nms=False):
    score = score[0, :, :]
    xy_text = np.argwhere(score > score_thresh)   ##位置坐标
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]  ##对坐标  按 h的从小到大的顺序进行排序
    valid_pos = xy_text[:, ::-1].copy()           ##  x,y 进行调换
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)

    if polys_restored.size == 0:
        return None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    start_time = time.time()
    if cv_nms:
        ## opencv-dnn-nms
        boxes_xywh = xyxy2xywh(boxes[:, :8])                                      ## xyxyxyxy --> xywh
        idxs = dnn.NMSBoxes(list(boxes_xywh.astype(np.int)), list(boxes[:, 8].astype(np.float)), score_thresh, nms_thresh)
        idxs = idxs.reshape(-1)
        boxes = boxes[idxs, :]

    else:
        ##  python-locality-nms
        boxes = nms_locality(boxes.astype(np.float64), nms_thresh)    ## return [ , 4*xy + score]

    print('nms time:{:.4f}'.format(time.time()-start_time))

    return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
    if boxes is None or boxes.size == 0:
        return None
    boxes[:, [0, 2, 4, 6]] /= ratio_w
    boxes[:, [1, 3, 5, 7]] /= ratio_h
    return np.around(boxes)


def detect(img, model, device, cv_nms=False):
    img, ratio_h, ratio_w = resize_img(img)
    start = time.time()
    with torch.no_grad():
        score, geo = model(load_pil(img).to(device))
    print('model time:{:.4f}'.format(time.time()-start))
    boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy(), nms_thresh = 0.2, cv_nms=cv_nms)
    return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img, boxes):
    if boxes is None:
        return img

    for box in boxes:
        cv.rectangle(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), (255, 0, 0), 2)
        #draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(255,0,0))
    return img


def model_predict(blob, net, label, candidate_num):
    net.setInput(blob)
    preds = net.forward()
    #print(preds)
    idx = np.argsort(preds[0])[::-1]
    text =[]
    Confidence =[]
    for i in range(candidate_num):
        text.append(label[idx[i]])
        Confidence.append(preds[0][idx[i]]*100)
    return text, Confidence


def recognition_word(img, boxes, rec_modelPath):
    deploy = os.path.join(rec_modelPath, 'deploy.prototxt')
    weights = os.path.join(rec_modelPath, 'weights.caffemodel')
    labelPath = os.path.join(rec_modelPath, 'label.txt')

    text_ = []
    Confidence_ = []


    row = open(labelPath, encoding='gbk').read().strip().split("\n")
    class_label = row

    net = dnn.readNetFromCaffe(deploy, weights)

    for box in boxes:
        img_word = img[int(box[1]):int(box[5]), int(box[0]):int(box[4])].copy()
        img_word = cv.cvtColor(img_word, cv.COLOR_RGB2GRAY)
        #img_word = img_word[:, :, 0]
        img_word = cv.resize(img_word, (64, 64))

        blob = dnn.blobFromImage(img_word, 1, (64, 64), (0.))
        text, Confidence = model_predict(blob, net, class_label, 1)

        text_.append(text[0])
        Confidence_.append(round(Confidence[0], 2))

    # print(text_)
    # print(Confidence_)
    return text_, Confidence_


def get_information(input_str):
    if "*" in input_str:
        return_information = "姓名： " + input_str[input_str.index('*')+8:] + ','
        return return_information

    elif "年" in input_str:
        return_information = "日期： " + input_str[:input_str.index('日')+1] + ','
        return return_information

    elif "站" in input_str:
        start_station = input_str[:input_str.index('站') + 1]
        train_number = "".join(re.findall(r'[A-Za-z0-9]', input_str))
        terminal_station = input_str[len(start_station+train_number):]
        return_information = "车次： " + train_number + ",起始： " + start_station + ",终点： " + terminal_station + ','
        return return_information

    elif "元" in input_str:
        return_information = "金额： " + input_str[1:input_str.index('元')+1] + ','
        return return_information


def reco_and_deal(boxes, img, rec_modelPath):
    key_words = ["*", "年", "站", "元"]  # 选择的关键字， 按顺序
    recogn_strs = []                 # 识别出来的全部字符

    id_y = np.argsort((boxes[:, 1]+boxes[:, 7])/2)       ## 按 字的中心点 y 进行升序
    boxes = boxes[id_y]

    mean_y = np.mean(boxes[:, 7] - boxes[:, 1]) / 2      ## 所有 box的 y 的平均值 的 一半
    boxes_centerY = (boxes[:, 7] + boxes[:, 1]) / 2

    cut_point = 0
    cut = [0]
    for b in np.arange(1, len(boxes)):
        if (boxes_centerY[b] - boxes_centerY[b-1]) >= mean_y*0.8:  ##  分界线
            sort_boxes = boxes[cut_point:b, :]
            boxes[cut_point:b, :] = sort_boxes[np.argsort(sort_boxes[:, 0])]
            cut_point = b
            cut.append(cut_point)
        if (b == len(boxes)-1) and cut_point != b:
            sort_boxes = boxes[cut_point:b+1, :]
            boxes[cut_point:b+1, :] = sort_boxes[np.argsort(sort_boxes[:, 0])]
            cut.append(b)

    # recognition
    text, conf = recognition_word(img, boxes, rec_modelPath)
    for i in range(len(cut)-1):                      ##分段放入 list中
        rstr = ''.join(text[cut[i]:cut[i+1]])
        recogn_strs.append(rstr)

    need_recogn_strs = ""
    for key_word in key_words:
        for recogn_str in recogn_strs:
            if key_word in recogn_str:
                need_recogn_strs += get_information(recogn_str)
                break

    return need_recogn_strs[:-1]


def me_main(img_path, use_linux=True):
    if use_linux:
        model_path = "./pths/model_epoch_fun100.pth"
        rec_modelPath = "/home/zjb/remote/EASTfp_noangle_newaugres/"
        res_img    = './res.bmp'
    else:
        model_path = r"F:\pycharm\PyCharm-professional2019.3.4\PycharmProjects\EASTfp_noangle_newaugres\pths\model_epoch_fun100.pth"
        rec_modelPath = r"F:\pycharm\PyCharm-professional2019.3.4\PycharmProjects\chinese6948_64"
        res_img = './res.bmp'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST_MobileV2(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # detect words
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    start_time1 = time.time()
    boxes = detect(img, model, device, cv_nms=True)
    print('total time:{:.4f}'.format(time.time() - start_time1))

    ## 对字符进行识别，并筛选 需要的 字符串 ， 返回 "ldka,ekek,dkdk"格式
    result_str = reco_and_deal(boxes, img, rec_modelPath)

    # plot_img = plot_boxes(img, boxes)
    # plot_img = cv.cvtColor(plot_img, cv.COLOR_RGB2BGR)
    # cv.imwrite("/home/zjb/remote/EASTfp_noangle_newaugres/receive_plot.jpg", plot_img)
    # cv2Image = Image.fromarray(plot_img)
    # cv2Image.show()

    return result_str


if __name__ == '__main__':
    img_path = r"C:\Users\Lenovo\Desktop\op\11.jpg"
    print(me_main(img_path, use_linux=False))


