import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
import math
import serial 
import time # Add this import for serial communication

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



def get_grid_cell(bbox_center_x, bbox_center_y, img_width, img_height, rows, cols):
    """Calculate which grid cell the bounding box center falls into."""
    cell_width = img_width / cols
    cell_height = img_height / rows
    grid_col = int(bbox_center_x // cell_width)
    grid_row = int(bbox_center_y // cell_height)
    return grid_row, grid_col

def draw_grid(image, rows, cols, color=(0, 255, 0), thickness=1):
    """Draw a grid on the image."""
    img_height, img_width, _ = image.shape
    cell_width = img_width // cols
    cell_height = img_height // rows
    for i in range(1, cols):
        cv2.line(image, (i * cell_width, 0), (i * cell_width, img_height), color, thickness)
    for i in range(1, rows):
        cv2.line(image, (0, i * cell_height), (img_width, i * cell_height), color, thickness)
    return image

    


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
        view_img=False,  # show results
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
):
    
    try:
        arduino_port = 'COM4'  # Update to the correct port if needed
        baud_rate = 9600
        ser = serial.Serial(arduino_port, baud_rate)
        time.sleep(2)  # Wait for the connection to initialize
        print("Serial connection established.")
    except serial.SerialException as e:
        print(f"Error: {e}")
        print("Could not open port. Please check if the port is correct or in use.")
        exit()
        
        
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
            s += '%gx%g' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Draw grid on the image
                rows, cols = 3, 3  # You can change the grid size (3x3, 4x4, etc.)
                im0 = draw_grid(im0, rows, cols)
                
                
            

                # Assuming 'det' contains the detection results, 'im0' is the image/frame from the webcam
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)  # Extract bounding box coordinates
                    bbox_center_x = (x1 + x2) / 2  # Center of bounding box
                    bbox_center_y = (y1 + y2) / 2
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1

                    # Define the origin at the center of the image (or any reference point)
                    origin_x = im0.shape[1] / 2  # Center x coordinate
                    origin_y = im0.shape[0] / 2  # Center y coordinate

                    # Calculate angles relative to the image center (origin)
                    theta_x = math.atan2(bbox_center_y - origin_y, bbox_center_x - origin_x)

                    # Convert angle to degrees and adjust to match servo motor control range
                    theta_x_degrees = math.degrees(theta_x)

                    # Normalize the angle to the range of the servo (0 to 180 degrees)
                    # Assuming -90 degrees (left edge) to +90 degrees (right edge) maps to 0-180 degrees for the servo
                    servo_angle = int((theta_x_degrees + 90) * (180 / 180))  # Normalize to 0-180

                    # Send this angle to the Arduino for servo control (assuming serial communication)
                    # arduino.write(f"{servo_angle}\n".encode())
                    
                    try:
                        ser.write(f"{theta_x_degrees}\n".encode())
                        print(f"Servo angle sent: {theta_x_degrees}")
                    except Exception as e:
                        print(f"Error sending data to Arduino: {e}")
                    
                    # Debug information
                    print(f'Detection falls into bounding box center: ({bbox_center_x}, {bbox_center_y})')
                    print(f'Servo angle for horizontal rotation: {servo_angle} degrees')

                    # Optionally mark the center of the bounding box in the frame
                    cv2.circle(im0, (int(bbox_center_x), int(bbox_center_y)), radius=5, color=(0, 0, 255), thickness=-1)  # Red dot at center

                    # Draw the bounding box and label
                    label = f'Fire! {conf:.2f}'
                    annotator.box_label(xyxy, label, color=(0, 0, 255))  # Red box for fire

                    # Print fire detection information with angles
                    print(f'Fire detected! Coordinates: Center=({bbox_center_x}, {bbox_center_y}), Width={bbox_width}, Height={bbox_height}')
                    print(f'Angle θx (degrees): {theta_x_degrees:.2f}, Servo angle: {servo_angle} degrees')
                    


                    # If you want to compute an estimated distance, use the formula:
                    # Distance = (Known Object Width * Focal Length) / Bounding Box Width
                    # Replace 'known_width' and 'focal_length' with appropriate values
                    known_width = 1.0  # Set the known width of the object (in meters or other unit)
                    focal_length = 1.0  # Set the camera's focal length
                    distance = (known_width * focal_length) / bbox_width
                    

                    print(f'Estimated distance to fire: {(distance*2300*1.5)-10} cm')

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    
                    
                    
                        
                        
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

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


        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({dt[1].dt * 1E3:.1f}ms)')

    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
        
        

    
    
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
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))  # Pass only opt, no FILE.stem
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
