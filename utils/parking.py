import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from utils.utils import write_log, fixBBox, get_degrees, xywhn2xyxy, letterbox
from utils.licenceText.licence import getlicence

normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])

class Predict_Object:
    def __init__(self, data_path, obj, GPSData) -> None:
        # TODO judgment Threshold
        self.Frames_THR = 0
        self.score_THR = 0

        self.image_path = None
        self.DateTime = None
        self.lat = None
        self.Lon = None
        self.isViolation = None
        self.cls = None
        self.id = None
        self.boxes = None
        self.FramesNum = 0
        self.score = 0.0
        self.conf = 0.0
        self.licence = None
        self.licence_confidence = 0
        self.update(data_path, obj, GPSData)
        

    
    def update(self, data_path, obj, GPSData=None) -> None:
        self.image_path = data_path
        self.DateTime = GPSData['DateTime']
        if GPSData != None:
            self.Lat = GPSData['Lat']
            self.Lon = GPSData['Lon']
        self.isViolation = obj['isViolation']

        self.cls = obj['cls']
        self.id = obj['id']
        self.boxes = obj['boxes']
        self.conf = obj['conf']

        self.FramesNum += 1
        self.score = self.score + (1 if self.isViolation else -1)*self.conf

        New_licence_confidence = obj['licence_confidence']
        if (self.licence!=None and len(self.licence) <=3) or self.licence_confidence <= New_licence_confidence:
            self.licence_confidence = New_licence_confidence
            self.licence = obj['licence']

    def is_Violation(self) -> bool:
        # TODO judgment Threshold
        return (self.FramesNum > self.Frames_THR) and (self.score > self.score_THR)


class Predict_Frame_Object:
    def __init__(self, ID:list, img:np.ndarray, LonLat = None) -> None:
        self.ID = ID
        self.img = img
        self.exitID = set()
        self.LonLat = [LonLat['Lon'], LonLat['Lat'],LonLat['direction']] if LonLat else None

    def update_preframe(self, ID:list, img:np.ndarray, LonLat = None) -> None:
        self.ID = ID
        self.img = img
        self.LonLat = [LonLat['Lon'], LonLat['Lat'],LonLat['direction']] if LonLat else None

    def update_exitID(self, seenID:int) -> None:
        if seenID != None:
            self.exitID.add(seenID)

def get_direction(preDirection: float, prePlace: list[float],
                  nowPlace: list[float]) -> int:
    """use preFrame location and Now location get facing direction

    Args:
        preDirection (float): pre facing direction
        prePlace (list[float]): [lon, lat, direction]
        nowPlace (list[float]): [lon, lat]

    Returns:
        int: facing direction
    """
    degrees = preDirection
    diff = ((nowPlace[0] - prePlace[0])*1E6,
            (nowPlace[1] - prePlace[1])*1E6)
    if pow(diff[0], 2) + pow(diff[1], 2) >= 200:
        degrees = get_degrees(diff)
    return int(degrees)


def check_disappear(predictor, PreFrame, NowID, trackID):
    save_dir = predictor.save_dir
    PreFrameID, exitID = PreFrame.ID, PreFrame.exitID
    plotted_img = PreFrame.img

    for PreID in PreFrameID:
        if PreID not in NowID and PreID not in exitID:
            PreFrame.update_exitID(PreID)
            licence = trackID[PreID].licence
            data ={
                    'imgpath':trackID[PreID].image_path,
                    'ID':int(PreID),
                    'licence':licence,
                    'cls':int(trackID[PreID].cls),
                    'isViolation':trackID[PreID].isViolation,
                    'box':trackID[PreID].boxes,
                    'GPS':{
                            'Lat':trackID[PreID].Lat, 
                            'Lon':trackID[PreID].Lon
                        },
                    }
            
            # 將是否為違停記錄下來
            if trackID[PreID].is_Violation():
                data['cls'] = 0
                isViolation = True
                cv2.imwrite(str(save_dir / 'Violation' / f'{PreID}.jpg'), plotted_img)
                msg = f'ID {PreID} ({licence}) is Violation'
            else :
                data['cls'] = 1
                isViolation = False
                msg = f'ID {PreID} ({licence}) is Legitimate'
            data['isViolation'] = isViolation
                
            if predictor.HaveGPS:
                location=[trackID[PreID].Lat, trackID[PreID].Lon]
                predictor.foliumMap.add_Marker(location,f'ID {PreID} ({licence})', isViolation)
                msg += f' at ({trackID[PreID].Lat}, {trackID[PreID].Lon})'

            del trackID[PreID]

            write_log(save_dir / 'result.txt', msg)
            predictor.json['data'].append(data)


def update_trackID(predictor, GPSData):
    boxes = predictor.results[0].boxes
    orig_img = predictor.results[0].orig_img
    data_path = predictor.results[0].path
    trackID = predictor.trackID
    NowID = [str(int(id)) for id in boxes.id] if boxes.id is not None else []
    PreFrame = update_trackID.PreFrame
    predictor.plotted_img = predictor.results[0].plot()

    image = cv2.cvtColor(orig_img.copy(), cv2.COLOR_BGR2RGB)
    drivable = predictor.batch[0][0].replace('images','drivable').replace('jpg','png')
    drivable = cv2.imread(drivable, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    drivable = cv2.cvtColor(drivable, cv2.COLOR_BGR2RGB)
    laneline = predictor.batch[0][0].replace('images','laneline').replace('jpg','png')
    laneline = cv2.imread(laneline, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    laneline = cv2.cvtColor(laneline, cv2.COLOR_BGR2RGB)

    # 由經緯度取的方向 
    GPSData['direction'] = None
    if predictor.HaveGPS and PreFrame.LonLat:
        GPSData['direction'] = get_direction(PreFrame.LonLat[2], PreFrame.LonLat, [GPSData['Lon'], GPSData['Lat']])
    
    for i in range(len(NowID)):
        box = fixBBox(boxes.xyxy[i], predictor.results[0].orig_shape)
        

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
        outputs = torch.argmax(conf, dim=1).tolist()[0]
        conf = conf.tolist()[0][outputs]
      


        # 取得車子在我的幾度角
        diff = [orig_img.shape[0] - int(boxes.xywh[i][1]), 
                int(boxes.xywh[i][0])-orig_img.shape[1]/2]
        car_degrees = get_degrees(diff)

        # 取得車牌號碼
        licence, confidence = getlicence(predictor, orig_img, box)
        obj = {
            'cls':str(int(boxes.cls[i])),
            'id': NowID[i],
            'boxes':box,
            'conf':float(boxes.conf[i]),
            'licence':licence,
            'licence_confidence':confidence,
            'isViolation': True if outputs==0 else False
        }

        # 將辨識到的做暫存
        if obj['id'] in trackID.keys():
            trackID[obj['id']].update(data_path, obj, GPSData)
        else:
            trackID[obj['id']] = Predict_Object(data_path, obj, GPSData)

    # 將消失的物件進行輸出
    cv2.imwrite(str(predictor.save_dir / 'result' / predictor.data_path.name), predictor.plotted_img)
    check_disappear(predictor, PreFrame, NowID, trackID)
    update_trackID.PreFrame.update_preframe(NowID, predictor.plotted_img, GPSData)

update_trackID.PreFrame = Predict_Frame_Object([],None,None)