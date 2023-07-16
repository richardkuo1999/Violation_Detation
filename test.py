import cv2
import time
import torch
import argparse
from pathlib import Path
from ultralytics import YOLO
import torchvision.transforms as transforms

from utils.Violation_Classification.models.model import build_model
from utils.Violation_Classification.utils.general import increment_path
from utils.utils import write_log, xywhn2xyxy, letterbox
from utils.Violation_Classification.utils.torch_utils import select_device
from utils.Violation_Classification.utils.plot import plot_one_box

normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])

classname = ['violations', 'legitimate']
colors = [(255,255,0), (0,255,255)]

def GTourMethod():
  def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./weights/traffic.pt', 
                        help='path to model file, i.e. yolov8n.pt, yolov8n.yaml') 
    parser.add_argument('--source', type=str, default='./inference/GH014155/val', 
                        help='path to data file or https://...')
    parser.add_argument('--device', default='', 
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--logDir', type=str, default='runs/test',
                            help='log directory')
    return parser.parse_args()
  args = parse_args()
  device = select_device(args.device)

  model = build_model(ch=[3,3], num_classes=2, tokensize=32,
                            split=False).to(device)
  # load weights
  checkpoint_file = './weights/Violation_Classification.pth'
  print("=> loading checkpoint '{}'".format(checkpoint_file))
  checkpoint = torch.load(checkpoint_file, map_location= device)
  model.load_state_dict(checkpoint['state_dict'])
  model = model.to(device).eval()
  source = Path(args.source)
  args.save_dir = Path(increment_path(Path(args.logDir)))
  args.save_dir.mkdir(parents=True, exist_ok=True)
  (args.save_dir/ 'show').mkdir(parents=True, exist_ok=True)
  save_dir = args.save_dir
  for image_path in (source / 'images') .glob('*.jpg'):
    image_path = str(image_path)
    txt_path = image_path.replace('images','labels').replace('jpg','txt')

    orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    image = cv2.cvtColor(orig_img.copy(), cv2.COLOR_BGR2RGB)

    plot_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    h, w = plot_image.shape[:2]

    with open(txt_path) as f:
      content = f.readlines()
    content = [x.strip() for x in content]
    boxes = []
    for line in content:
        # obj_id, x_c_n, y_c_n, width_n, height_n = line.split()
        data = line.split()
        x_c_n, y_c_n, width_n, height_n = data[1:5]
        boxes.append([float(x_c_n), float(y_c_n), float(width_n), float(height_n)])


    drivable = image_path.replace('images','drivable').replace('jpg','png')
    drivable = cv2.imread(drivable, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    drivable = cv2.cvtColor(drivable, cv2.COLOR_BGR2RGB)
    laneline = image_path.replace('images','laneline').replace('jpg','png')
    laneline = cv2.imread(laneline, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    laneline = cv2.cvtColor(laneline, cv2.COLOR_BGR2RGB)

    
    for i in range(len(boxes)):
        ori_bbox = boxes[i].copy()
        bbox = torch.FloatTensor(boxes[i])
        obox = bbox.clone()
        bbox[2], bbox[3] = bbox[2]*2, bbox[3]*2
        object_xyxy = xywhn2xyxy(bbox,image.shape[:2])
        cut_image = image[object_xyxy[1]:object_xyxy[3],
                        object_xyxy[0]:object_xyxy[2]]
        cut_drivable = drivable[object_xyxy[1]:object_xyxy[3],
                        object_xyxy[0]:object_xyxy[2]]
        cut_laneline = laneline[object_xyxy[1]:object_xyxy[3],
                        object_xyxy[0]:object_xyxy[2]]
        

        (cut_image, cut_drivable, cut_laneline), ratio, pad = letterbox((cut_image, cut_drivable, cut_laneline),\
                                          (256,256), auto=False)
        
        cut_image = transform(cut_image).unsqueeze(0).to(device)

        cut_drivable = transforms.ToTensor()(cut_drivable).unsqueeze(0).to(device)
        cut_laneline = transforms.ToTensor()(cut_laneline).unsqueeze(0).to(device)
        bbox = bbox.unsqueeze(0).to(device)
        outputs = model(cut_image, cut_laneline, cut_drivable, bbox)
        conf = torch.softmax(outputs, dim=1)
        cls = torch.argmax(conf, dim=1).tolist()[0]
        conf = conf.tolist()[0][cls]
        xyxy = xywhn2xyxy(obox, (h, w))
        plot_one_box(xyxy, plot_image , label=classname[cls], color=colors[int(cls)], line_thickness=4)
        # predictor.results[0].boxes.cls[i] = outputs
        # predictor.results[0].boxes.conf[i] = conf
        msg = f'{cls} {ori_bbox[0]} {ori_bbox[1]} {ori_bbox[2]} {ori_bbox[3]} {conf}'
        write_log(save_dir/Path(txt_path).name, msg)
    cv2.imwrite(str(save_dir / 'show' / Path(image_path).name), plot_image)



def ourMethod():

  def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./weights/traffic.pt', 
                        help='path to model file, i.e. yolov8n.pt, yolov8n.yaml') 
    parser.add_argument('--source', type=str, default='./inference/GH014155/val/images', 
                        help='path to data file or https://...')
    parser.add_argument('--data', type=str, default='F:/ITRI/YOLOv8/ultralytics/datasets/class2.yaml', 
                help='path to data file, i.e. coco128.yaml')
    parser.add_argument('--device', default='', 
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()
  
  def on_predict_start(predictor):
    predictor.source = str(Path(predictor.dataset.files[0]).parent.parent)
    (predictor.save_dir/ 'label').mkdir(parents=True, exist_ok=True)
    (predictor.save_dir/ 'show').mkdir(parents=True, exist_ok=True)
    
    predictor.Violation = build_model(ch=[3,3], num_classes=2, tokensize=32,
                            split=False).to(predictor.device)
    # load weights
    checkpoint_file = './weights/Violation_Classification.pth'
    print("=> loading checkpoint '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location= predictor.device)
    predictor.Violation.load_state_dict(checkpoint['state_dict'])
    predictor.Violation = predictor.Violation.to(predictor.device).eval()


  def on_predict_batch_end(predictor):
    boxes = predictor.results[0].boxes
    orig_img = predictor.results[0].orig_img
    save_dir = predictor.save_dir
    image_path = predictor.results[0].path.split('\\')[-1]
    txt_path = save_dir / 'label' / image_path.replace('jpg','txt')

    image = cv2.cvtColor(orig_img.copy(), cv2.COLOR_BGR2RGB)
    drivable = predictor.batch[0][0].replace('images','drivable').replace('jpg','png')
    drivable = cv2.imread(drivable, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    drivable = cv2.cvtColor(drivable, cv2.COLOR_BGR2RGB)
    laneline = predictor.batch[0][0].replace('images','laneline').replace('jpg','png')
    laneline = cv2.imread(laneline, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    laneline = cv2.cvtColor(laneline, cv2.COLOR_BGR2RGB)

    plot_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    h, w = plot_image.shape[:2]

    for i in range(len(boxes)):
        ori_bbox = boxes.xywhn[i].clone().tolist()
        bbox = boxes.xywhn[i].clone()
        bbox[2], bbox[3] = bbox[2]*2, bbox[3]*2
        object_xyxy = xywhn2xyxy(bbox,boxes.orig_shape)
        cut_image = image[object_xyxy[1]:object_xyxy[3],
                        object_xyxy[0]:object_xyxy[2]]
        cut_drivable = drivable[object_xyxy[1]:object_xyxy[3],
                        object_xyxy[0]:object_xyxy[2]]
        cut_laneline = laneline[object_xyxy[1]:object_xyxy[3],
                        object_xyxy[0]:object_xyxy[2]]
        

        (cut_image, cut_drivable, cut_laneline), ratio, pad = letterbox((cut_image, cut_drivable, cut_laneline),\
                                          (256,256), auto=False)
        
        cut_image = transform(cut_image).unsqueeze(0).to(predictor.device)

        cut_drivable = transforms.ToTensor()(cut_drivable).unsqueeze(0).to(predictor.device)
        cut_laneline = transforms.ToTensor()(cut_laneline).unsqueeze(0).to(predictor.device)
        bbox = bbox.unsqueeze(0).to(predictor.device)
        outputs = predictor.Violation(cut_image, cut_laneline, cut_drivable, bbox)
        conf = torch.softmax(outputs, dim=1)
        cls = torch.argmax(conf, dim=1).tolist()[0]
        conf = conf.tolist()[0][cls]

        plot_one_box(boxes.xyxy[i], plot_image , label=classname[cls], color=colors[int(cls)], line_thickness=4)
        predictor.results[0].boxes.cls[i] = cls
        predictor.results[0].boxes.conf[i] = conf
        msg = f'{cls} {ori_bbox[0]} {ori_bbox[1]} {ori_bbox[2]} {ori_bbox[3]} {conf}'
        write_log(txt_path, msg)
    cv2.imwrite(str(save_dir / 'show' / Path(image_path).name), plot_image)

  args = parse_args()

  # Load a model
  model = YOLO(args.model) # load a custom model

  # Add the custom callback to the model
  model.add_callback("on_predict_start", on_predict_start)
  model.add_callback("on_predict_batch_end", on_predict_batch_end)

  # Iterate through the results and frames
  model.predict(source=args.source)
  
  

def onlyYOLOv8():

  def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./weights/2class.pt', 
                        help='path to model file, i.e. yolov8n.pt, yolov8n.yaml')
    parser.add_argument('--source', type=str, default='./inference/GH014155/val/images', 
                    help='path to data file or https://...')
    parser.add_argument('--data', type=str, default='F:/ITRI/YOLOv8/ultralytics/datasets/class2.yaml', 
                    help='path to data file, i.e. coco128.yaml')
    return parser.parse_args()
  
  def on_predict_start(predictor):
    (predictor.save_dir/ 'show').mkdir(parents=True, exist_ok=True)

  def on_predict_batch_end(predictor):
    boxes = predictor.results[0].boxes
    orig_img = predictor.results[0].orig_img
    save_dir = predictor.save_dir
    image_path = predictor.results[0].path
    plot_image = predictor.results[0].orig_img.copy()
    for i in range(len(boxes)):
      cls = int(boxes.cls[i])
      plot_one_box(boxes.xyxy[i], plot_image , label=classname[cls], color=colors[int(cls)], line_thickness=4)
    cv2.imwrite(str(save_dir / 'show' / Path(image_path).name), plot_image)
  args = parse_args()
  # Load a model
  model = YOLO(args.model)  # load a pretrained model (recommended for training)
 
  # Add the custom callback to the model
  model.add_callback("on_predict_start", on_predict_start)
  model.add_callback("on_predict_batch_end", on_predict_batch_end)

  # or you can set the data you want to val
  model.predict(source=args.source, save_txt=True, save_conf=True)

if __name__=='__main__':
  # onlyYOLOv8()
  # ourMethod()
  GTourMethod()