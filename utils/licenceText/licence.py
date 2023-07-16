import cv2
import torch
# import easyocr
from PIL import Image
from utils.licenceText.strhub.data.module import SceneTextDataModule

from utils.plot import box_label

# Load model and image transforms
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

# reader = easyocr.Reader(['en'], gpu=True)

def ocr_image(img):
    # gray = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
    result = reader.readtext(img)
    text = ""

    for res in result:
        if len(result) == 1:
            text = res[1]
        if len(result) >1 and len(res[1])>6 and res[2]> 0.2:
            text = res[1]
        # text += res[1] + " "
    return str(text)


def getText(img):
    
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
    img = img_transform(img).unsqueeze(0)

    logits = parseq(img)
    logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

    # Greedy decoding
    pred = logits.softmax(-1)
    label, confidence = parseq.tokenizer.decode(pred)
    print('Decoded label = {}'.format(label[0]))

    return str(label[0]), float(confidence[0][-1])


def getlicence(predictor, img, box):
    cx1,cy1,cx2,cy2 = list(map(int,box))
    img_car = img[cy1:cy2,cx1:cx2]
    results = predictor.licenceDetector(source=img_car,
                                        device=predictor.device)[0]

    text = None
    confidence = 0
    for box in results.boxes.xyxy:
        box = [int(a) for a in box]
        lx1,ly1,lx2, ly2 = int(box[0]), int(box[1]), int(box[2]),int(box[3])
        img_l = img_car[ly1:ly2,lx1:lx2]
        
        text, confidence = getText(img_l.copy())

        box[0]+=cx1
        box[1]+=cy1
        box[2]+=cx1
        box[3]+=cy1
        box_label(predictor.plotted_img, box, text, line_width=5)
    return text, confidence
