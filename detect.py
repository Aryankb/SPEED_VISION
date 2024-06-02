import argparse
import os
import platform
import sys
from pathlib import Path
import torch
import time
from motpy import Detection, MultiObjectTracker
from collections import namedtuple
import cv2
import pandas as pd
from ultralytics import YOLO
import math
from collections import Counter,defaultdict 







FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode





from paddleocr import PaddleOCR
import cv2
ocr = PaddleOCR(lang='en') # need to run only once to load model into memory

def perform_ocr_on_image(img,coord):
    x,y,w,h=map(int,coord)
    H, W, _ = img.shape


    # Ensure bounding box coordinates are within image dimensions
    if x < 0 or y < 0 or w > W or h>H:
        print(f"Warning: Bounding box {coord} is out of image bounds.")
        return ""
    cropped_img=img[y:h,x:w]
    if cropped_img.size == 0:
        print(f"Warning: Cropped image for bounding box {coord} is empty.")
        return ""
    

    txt=""
    
    result = ocr.ocr(cropped_img, det=False, cls=False)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            txt+=line[0]
    return str(txt)

def calculate_fps(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get the FPS (frames per second)
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Release the video capture object
    video.release()
    
    return fps
def distance_from_line(cx, cy, sx, sy, ex, ey):
    if ex - sx != 0:
        slope = (ey - sy) / (ex - sx)
        constant = sy - slope * sx
        return abs((slope * cx - cy + constant)) / math.sqrt(slope**2 + 1)
    else:
        return abs(cx - sx)
def calculate_intersection(rect1, rect2):
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2
    
    # Calculate the coordinates of the intersection rectangle
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area


@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        linecoordss=ROOT,
        threshold=ROOT,
        DISTAN=ROOT
):
    
    fPs=calculate_fps(source)
    l1sx,l1sy,l1ex,l1ey,l2sx,l2sy,l2ex,l2ey=linecoordss
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # Initialize the YOLO model
    modelll = YOLO('yolov8s.pt')

    # Define the class list
    class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    # Initialize variables
    count = 0
    up = {}
    counter_up = []
    speeD = defaultdict(lambda:[0,[]])

    # Initialize the motpy tracker
    tracker = MultiObjectTracker(dt=1/fPs)  # You may need to adjust the dt parameter based on your frame rate
    FrameS=1
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)






        

        

        # Define the distance between the lines (in meters)
        # DISTAN = 10

        # Function to calculate distance from a point to a line
        

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            original_shape=im0.shape
            count += 1
            im0 = cv2.resize(im0, (1020, 500))

            results = modelll.predict(im0)
            a = results[0].boxes.data
            a = a.detach().cpu().numpy()
            px = pd.DataFrame(a).astype("float")
            detections = []

            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                d = int(row[5])
                c = class_list[d]
                if 'car' in c or 'motorcycle' in c or 'bus' in c or 'truck' in c:
                    detections.append(Detection(box=[x1, y1, x2, y2]))

            tracker.step(detections)
            tracked_objects = tracker.active_tracks()

            for obj in tracked_objects:
                x3, y3, x4, y4 = obj.box
                cx = (int(x3) + int(x4)) // 2
                cy = (int(y3) + int(y4)) // 2
                id = obj.id
                x3=int(x3)
                y3=int(y3)
                x4=int(x4)
                y4=int(y4)
                if int(x3) and int(y3) and int(x4) and int(y4):
                    try:
                        cv2.rectangle(im0, (int(x3), int(y3)), (int(x4), int(y4)), (255, 255, 0), 2)  # Draw a blue box
                    except:
                        pass
                    cv2.circle(im0, (cx, cy), 4, (0, 0, 255), -1)
                    # Calculate distances from the lines
                    distance1 = distance_from_line(cx, cy, l1sx, l1sy, l1ex, l1ey)
                    distance2 = distance_from_line(cx, cy, l2sx, l2sy, l2ex, l2ey)

                    if distance1 < 6:
                        up[id] = FrameS
                        cv2.circle(im0, (cx, cy), 4, (0, 255, 0), -1)
                    if id in up:
                        if id not in speeD:
                            cv2.circle(im0, (cx, cy), 4, (0, 255, 0), -1)
                        if distance2 < 6:
                            elapsed1_time = (FrameS - up[id])/fPs
                            cv2.circle(im0, (cx, cy), 4, (0, 0, 255), -1)
                            if counter_up.count(id) == 0:
                                counter_up.append(id)
                                a_speed_ms1 = DISTAN / elapsed1_time
                                a_speed_kh1 = a_speed_ms1 * 3.6
                                
                                cv2.rectangle(im0, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 2)  # Draw bounding box
                                cv2.putText(im0, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                                cv2.putText(im0, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                                speeD[id][0] = a_speed_kh1


            text_color = (0, 0, 0)  # Black color for text
            yellow_color = (0, 255, 255)  # Yellow color for background
            red_color = (0, 0, 255)  # Red color for lines
            blue_color = (255, 0, 0)  # Blue color for lines

            cv2.line(im0, (l1sx, l1sy), (l1ex, l1ey), red_color, 2)
            cv2.putText(im0, ('1st Line'), (l1sx, l1sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            cv2.line(im0, (l2sx, l2sy), (l2ex, l2ey), blue_color, 2)
            cv2.putText(im0, ('2nd Line'), (l2sx, l2sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            cv2.rectangle(im0, (0, 0), (250, 90), yellow_color, -1)
            cv2.putText(im0, ('COUNT - ' + str(len(counter_up))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            


            ######################################################################################################
            im0=cv2.resize(im0,(original_shape[1],original_shape[0]))
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))












            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                        #######################################################################
                        #ocr


                        text_ocr=perform_ocr_on_image(im0,xyxy)
                        label=text_ocr
                        maxi_id=""
                        for bbox in tracked_objects:
                            x3, y3, x4, y4 = bbox.box
                            id=bbox.id
                            x,y,w,h=map(int,xyxy)
                            rect1=[x3,y3,x4,y4]
                            rect2=[x,y,w,h]
                            intersection=calculate_intersection(rect1,rect2)
                            if intersection==abs(x-w)*abs(y-h):
                                maxi_id=id
                                break
                        if maxi_id!="":
                            
                            speeD[maxi_id][1].append(label)
                                
                        











                        annotator.box_label(xyxy, label, color=colors(c, True))

                        
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)



                        

            # Stream results
            
            im0 = annotator.result()
            cv2.imshow('Frame', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # if view_img:
            #     if platform.system() == 'Linux' and p not in windows:
            #         windows.append(p)
            #         cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            #     cv2.imshow(str(p), im0)
            #     cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        
        FrameS+=1

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)



    import csv
    with open(project+'results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["speed","liscence","overspeeding"])
        for speed_lisc in speeD.values():
            try:
                if int(speed_lisc[0])>threshold:
                
                    writer.writerow([speed_lisc[0],Counter(speed_lisc[1]).most_common(1)[0][0],"Y"])
                else:
                    writer.writerow([speed_lisc[0],Counter(speed_lisc[1]).most_common(1)[0][0],"N"])

            except:
                pass



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--linecoordss', nargs='+', type=int, default=[10, 500, 500, 500, 10, 250, 250, 250], help='Coordinates of the lines')
    parser.add_argument('--threshold', type=int, default=10, help='Threshold value')
    parser.add_argument('--DISTAN', type=int, default=10, help='Distance value')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    
    opt = parse_opt()
    main(opt)
