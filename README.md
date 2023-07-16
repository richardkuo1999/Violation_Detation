# Multi-task-learning-for-road-perception

## Datasets
  - vehicle detection: [BDD100K](https://www.vis.xyz/bdd100k/)
  - Illegal parking detection: [Our dataset](./dataset/)

## Weight
  Download from here [vehicle_Detection](https://drive.google.com/file/d/19mJ_e6KvvEEpo0Ah03NwQibzIj8Y6CHS/view?usp=sharing)
                     ,[licence_Detection](https://drive.google.com/file/d/1k2AE2miIOKeGWO1ncExy6D7Hw1s-WQ1d/view?usp=sharing)
                     ,[Violation_Classification](https://drive.google.com/file/d/1qaYYTDYY_wXv5YnRQ6MbrWRLeyv4CVVA/view?usp=drive_link)<br>
  Put that Weight to [here](./weights/)
  
## Requirement
  This codebase has been developed with
  ```
    Python 3.9
    Cuda 12
    Pytorch 2.0.1
  ```
  See requirements.txt for additional dependencies and version requirements.
  ```shell
    pip install -r requirements.txt
  ```

## main command
  You can change the data use, Path, and Classes, and merge some classes from [here](/data).

  ### Train
  Train vehicle detection 
  ```shell
  python test_python.py
  ```
  Train Illegal parking classification
  - see [here](https://github.com/richardkuo1999/Violation_Classification)
  ### Test
  Put the weight of vehicle detection(rename to traffic.pt) and Illegal parking classification(rename to Violation_Classification.pth) into [here](./weights).
  ```shell
  python test.py
  ```
  ### Predict
  Put the weight of vehicle detection(rename to traffic.pt) and Illegal parking classification(rename to Violation_Classification.pth) into [here](./weights).
  <br>
  licence number identify weight also put [here](./weights). Download from [here](https://github.com/baudm/parseq) and [here](https://github.com/shihyung/Yolov4_car_plate_detection_recognition)(rename to licence.pt).
  ```shell
  python main.py
  ```
  ### Tensorboard
  ```shell
    tensorboard --logdir=runs
  ```

## Argument
  ### Train vehicle detection
  | Source           |   Argument                  |     Type    | Notes                                                                        |
  | :---             |    :----:                   |     :----:  |   ---:                                                                       |
  | data             | 'data/multi.yaml'           | str         | dataset yaml path                                                            |
  | epochs           | 500                         | int         | number of epochs to train for                                                |
  | batch            | 16                          | int         | number of images per batch                                                   |
  | workers          | 6                           | int         | maximum number of dataloader workers                                         |
  | device           | ''                          | None        | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu         |
  ### Test
  <!-- | Source           |   Argument                  |     Type    | Notes                                                                        |
  | :---             |    :----:                   |     :----:  |   ---:                                                                       |
  | hyp              | 'hyp/hyp.scratch.yolop.yaml'| str         | hyperparameter path                                                          |
  | DoOneHot         | False                       | bool        | do one hot or not                                                            |
  | useSplitModel    | False                       | bool        | use multi resnet do feature extract                                          |
  | tokensize        | 32                          | int         | size of the tokens                                                           |
  | data             | 'data/multi.yaml'           | str         | dataset yaml path                                                            |
  | weights          | './weights/epoch-200.pth'   | str         | model.pth path(s)                                                            |
  | logDir           | 'runs/train'                | str         | log directory                                                                |
  | batch_size       | 15                          | int         | 	number of images per batch                                                  |
  | workers          | 6                           | int         | maximum number of dataloader workers                                         |
  | device           | ''                          | None        | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu         | -->
  ```
    I believe in you.
    I have faith in you.
    You got this!
    You are almost there!
  ```
  ### Predict
  | Source           |   Argument                  |     Type    | Notes                                                                        |
  | :---             |    :----:                   |     :----:  |   ---:                                                                       |
  | model              | './weights/traffic.pt'    | str         | vehicle detection weight path                                                |
  | source           | './inference/val'           | str         | inference file path                                                          |
  | device           | ''                          | None        | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu         |