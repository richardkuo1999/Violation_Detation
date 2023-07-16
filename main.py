import time
import argparse
from ultralytics import YOLO


from utils.callback import on_predict_start, on_predict_batch_end
from utils.utils import save2json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./weights/traffic.pt', 
                        help='path to model file, i.e. yolov8n.pt, yolov8n.yaml')
    # parser.add_argument('--source', type=str, default='https://www.youtube.com/watch?v=GKNZbiXMGXw', 
    parser.add_argument('--source', type=str, default='./inference/GH014155/images', 
                        help='path to data file or https://...')
    parser.add_argument('--device', default='', 
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Load a model
    model = YOLO(args.model) # load a custom model

    # Add the custom callback to the model
    model.add_callback("on_predict_start", on_predict_start)
    model.add_callback("on_predict_batch_end", on_predict_batch_end)

    # Iterate through the results and frames
    start_t = time.time()
    results =  model.track(source=args.source, show=False,
                           save=False, save_txt=False)
    totalTime = time.time()-start_t
    

    save_path = model.predictor.save_dir
    # save to json
    save2json(model.predictor.json, save_path)
    # show on the map
    if model.predictor.HaveGPS:
        model.predictor.foliumMap.map_save(save_path/'parking.html')

    print(f'Total time is {totalTime}, Average time is {totalTime/len(results)}, FPS is {len(results)/totalTime}')


