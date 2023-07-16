import torch
from pathlib import Path
from ultralytics import YOLO

from utils.parking import update_trackID
from utils.utils import write_log, loadGPS
from utils.map import Map
from utils.Violation_Classification.models.model import build_model


def on_predict_start(predictor):
    predictor.source = str(Path(predictor.dataset.files[0]).parent.parent)
    predictor.MapData = loadGPS(predictor.source)
    predictor.trackID = {}
    predictor.HaveGPS = len(predictor.MapData) > 0
    (predictor.save_dir / 'Violation').mkdir(parents=True, exist_ok=True)
    (predictor.save_dir / 'result').mkdir(parents=True, exist_ok=True)
    predictor.licenceDetector = YOLO('./weights/licence.pt')

    predictor.Violation = build_model(ch=[3,3], num_classes=2, tokensize=32,
                            split=False).to(predictor.device)
    # load weights
    checkpoint_file = './weights/Violation_Classification.pth'
    print("=> loading checkpoint '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location= predictor.device)
    predictor.Violation.load_state_dict(checkpoint['state_dict'])
    predictor.Violation = predictor.Violation.to(predictor.device).eval()

    predictor.json = {'data':[]}
    if predictor.HaveGPS:
        predictor.foliumMap = Map()



def on_predict_batch_end(predictor):
    GPS_data = None
    
    if predictor.HaveGPS:
        GPS_data = {
            'Lat': float(predictor.MapData[predictor.seen-1]['GPS']['Lat']),
            'Lon': float(predictor.MapData[predictor.seen-1]['GPS']['Lon']),
            'isViolation': predictor.MapData[predictor.seen-1]['isDanger'],
            'DateTime': None,
        }

    update_trackID(predictor, GPS_data)
    write_log(predictor.save_dir / 'result.txt', '--------------------------------')
