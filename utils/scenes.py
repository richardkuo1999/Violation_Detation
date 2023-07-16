

def RiskLevel(predictor, orig_img):
    # classify
    result = predictor.scenesCls(orig_img)
    
    score = result[0].probs.tolist()
    idx = score.index(max(score))
    return result[0].names[idx] == 'danger'

    # carNum = len(predictor.results[0].boxes.cls)
    # return carNum > 5