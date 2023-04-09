import argparse
import time
from pathlib import Path
import glob
import os
import csv
import cv2

import numpy as np
# from utils.datasets import LoadImages

import tensorflow as tf
def letterbox(combination, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    
    img, seg = combination if len(combination) == 2 else (combination, None)
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        if seg:
            seg = cv2.resize(seg, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if seg is not None:  
        seg = cv2.copyMakeBorder(seg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)  # add border
    return (img, seg), ratio, (dw, dh)

class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            _, extension = os.path.splitext(p)
            if extension == '.txt':
                with open(p, 'r') as file_txt:
                    files = [line.strip() for line in file_txt]
            else:
                files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')
        img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        ni = len(images)
        
        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = ni # number of files
        self.mode = 'image'

        self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        assert img0 is not None, 'Image Not Found ' + path
        

        # Padded resize
        (img, seg), ratio, pad = letterbox(img0, self.img_size, stride=self.stride)
        h0, w0 = img0.shape[:2]
        h, w = img.shape[:2]
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, shapes

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_shape[0])  # y1, y2

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, agnostic=False):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections

    t = time.time()
    output = [tf.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        box = xywh2xyxy(x[:, :4]) # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        # Detections matrix nx6 (xyxy, conf, cls)
        # best class only
        conf, j = np.max(x[:, 5:], axis=1, keepdims=True), np.argmax(x[:,5:], axis=1, keepdims=True)
        x = np.concatenate((box, conf, j.astype(np.float32)), 1)[conf.flatten() > conf_thres]
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = tf.image.non_max_suppression(boxes = boxes, scores = scores, max_output_size=x.shape[0], iou_threshold = iou_thres)# NMS

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def detect():
    source, weights, imgsz = opt.source, opt.weights, opt.img_size
    print(weights)
    # Directories
    if opt.task == 'detection':
        save_dir = Path(opt.output, exist_ok=True) / 'object_detections'
    elif opt.task == 'segmentation':
        save_dir = Path(opt.output, exist_ok=True) / 'segmentation'
    else:
        print(f'your task is {opt.task}, please choose detection or segmentation.')
        exit()
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    
    stride = 32  # model stride
    imgsz = 640  # check img_size
    
    print('Load Image')
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    #names = model.module.names if hasattr(model, 'module') else model.names
    obj_names = ['vehicle', 'pedestrian', 'scooter', 'bicycle']
    seg_names = ['background','drivable area', 'alternative', 'single', 'double','dashed']

    device = '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'

    with tf.device(device):
        # Load model
        model = tf.lite.Interpreter(model_path=weights) #load model
        model.allocate_tensors() #分配内存
        obj_save_list = []
        for path, img, im0s, vid_cap, shapes in dataset:

            input_details = model.get_input_details()
            output_details = model.get_output_details()

            h, w = img.shape[-2:]
            dh, dw = (imgsz - h) //2, (imgsz-w) // 2
            img = cv2.copyMakeBorder(np.transpose(img, (1, 2, 0)), dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            img = np.transpose(img, (2, 0, 1))
            img = tf.convert_to_tensor(img, dtype=tf.float32)
            img /= 255.0
            if img.shape.ndims == 3:
                img = tf.expand_dims(img, axis=0)
            model.set_tensor(input_details[0]['index'], img) #將圖片填充到輸入張量中
            model.invoke()  #推理
            # output_data_0 = model.get_tensor(output_details[0]['index']) #(3, 40, 40, 9)
            # output_data_3 = model.get_tensor(output_details[3]['index']) #(3, 20, 20, 9)
            # output_data_4 = model.get_tensor(output_details[4]['index']) #(3, 80, 80, 9)
            p = Path(path)  # to Path
            if opt.task == 'segmentation':
                # mask
                seg_pred = model.get_tensor(output_details[1]['index']) #(6, 160, 160) -> segmentation
                height, width = shapes[0]
                seg_pred = np.transpose(seg_pred, (0, 2, 3, 1)) #(1,6,160,160) --> (1,160,160,6)   #tf.resize會針對height,width做更改(batch, height, width, channels) 因此先改順序
                seg_pred = tf.image.resize(seg_pred, size=(imgsz, imgsz), method=tf.image.ResizeMethod.BILINEAR) #插值到640,640  (1,640,640,6)
                seg_pred = seg_pred[:, dh:(imgsz-dh), dw:(imgsz-dw),:] #前面padding是將原圖片(384,640)貼到640,640的0矩陣，使用0:h,0:w的方式padding，因此這邊使用:h,:w只保留原圖的384,640
                seg_mask = tf.image.resize(seg_pred, size=(height, width), method=tf.image.ResizeMethod.BILINEAR)
                seg_mask = np.transpose(seg_mask.numpy(), (0, 3, 1, 2)) #(1,720,1280,6) --> (1,6,720,1280)
                # 取最大值索引
                seg_mask = tf.math.argmax(seg_mask, axis=1)
                seg_mask = np.squeeze(seg_mask.numpy())
                assert seg_mask.shape[0] != 1, f'seg_mask {seg_mask.shape}'
                assert np.all(seg_mask <= 5), f'seg_mask bigger then 5 '

                save_seg_path = str(save_dir / (f'{p.name.split(".")[0]}.png'))
                cv2.imwrite(save_seg_path, seg_mask)
            elif opt.task == 'detection':
                det_pred = model.get_tensor(output_details[2]['index']) #(25200, 9) -> detection
                # Apply NMS
                det_pred = non_max_suppression(det_pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)
                #gn = np.array(im0s.shape)[[1,0,1,0]] # normalization gain whwh
                for i, det in enumerate(det_pred):
                    if (len(det)):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                        xywh_list = xyxy2xywh(det[:,:4]).astype(np.int32).tolist() # xywh
                        conf_list = det[:, 4:5].tolist()
                        cls_list = (det[:, 5:] + 1).astype(np.int32).tolist() # cls
                        
                        for xywh, conf, cls in zip(xywh_list, conf_list, cls_list):
                            obj_save_list.append([str(p.name), *cls, *xywh, *conf])
                    
                        # for *xyxy, conf, cls in reversed(det):
                        #     names = ['vehicle', 'pedestrian', 'scooter', 'bicycle']
                        #     label = f'{names[int(cls)]} {conf:.2f}'
                        #     colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
                        #     plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=1)
                        #     cv2.imwrite(str(save_dir / p.name), im0s)
        
    if opt.task == 'detection':
        with open(str(save_dir / 'submission.csv'), 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(['image_filename', 'label_id', 'x', 'y', 'w', 'h', 'confidence'])
            write.writerows(obj_save_list)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['detection','segmentation'], help='detection/segmentation')
    parser.add_argument('source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('output', type=str, help='output path')
    parser.add_argument('--weights', type=str, default='weights/best.tflite', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    detect()