import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='weights/yolov8m.pt', 
                        help='path to model file, i.e. yolov8n.pt, yolov8n.yaml')
    parser.add_argument('--data', type=str, default='./ultralytics/datasets/BDDcar.yaml', 
                        help='path to data file, i.e. coco128.yaml')
    parser.add_argument('--epochs', type=int, default=500, 
                        help='number of epochs to train for')
    parser.add_argument('--patience', type=int, default=200, 
                        help='epochs to wait for no observable improvement for early stopping of training')
    parser.add_argument('--batch', type=int, default=8, 
                        help='number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--imgsz', type=int, default=640, 
                        help='size of input images as integer or w,h')
    parser.add_argument('--device', type=str, default='0', 
                        help='device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu')
    parser.add_argument('--workers', type=int, default=4, 
                        help='number of worker threads for data loading (per RANK if DDP)')
    parser.add_argument('--resume', type=bool, default=False, 
                        help='resume training from last checkpoint')
    return parser.parse_args()


if __name__=='__main__':
  args = parse_args()

  # Load a model
  model = YOLO(args.model) # load a pretrained model (recommended for training)
  
  # Train the model
  model.train(mode="detect", data=args.data,
                epochs=args.epochs, patience=args.patience,
                batch=args.batch, workers=args.workers,
                device=args.device, resume=args.resume,
                imgsz=args.imgsz)