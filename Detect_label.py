import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./runs/detect/laneline/weights/best.pt', 
                        help='path to model file, i.e. yolov8n.pt, yolov8n.yaml')
    parser.add_argument('--source', type=str, default='F:/tool/get_GPS_and_images/output/GH024152', 
                        help='path to data file')
    parser.add_argument('--imgsz', type=int, default=640, 
                        help='size of input images as integer or w,h')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Load a model
    model = YOLO(args.model) # load a custom model

    model.predict(source=args.source, 
                  imgsz=args.imgsz, conf=0.5,
                  save=True, save_txt=True)