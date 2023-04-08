

## Requirements

python 3.9 and all requirements.txt dependencies installed. To install run:
<details><summary> <b>Expand</b> </summary>

``` shell
python -m pip install -U pip
pip install -r requirements.txt
```

</details>

## Probject Structure
```python
.
└──MMTSNET
   ├──data               # inference images
   ├──weights            # model weight
   │  ├──best.pb
   │  │  └──...
   │  └──best.tflite 
   ├──README.md
   ├──requirements.txt   # additional dependencies and version requirements
   └──run_model.py
```

## Data Structure
We recommend the dataset directory structure to be the following:
```
.
└──data
   ├──Public_Private_Testing_Dataset
   │  ├──0001.jpg
   │  ├──0002.jpg
   │  ├──...
   │  └──0700.jpg
   └──Public_Private_Testing_Dataset_Only_for_detection
      ├──itp_1.jpg
      ├──itp_2.jpg
      ├──...
      └──itpq_2700.jpg
```

## Inference
The first argument is task name, you can choose to do either `detection` or `segmentation`. The second argument is data source, where you can put any images you want to infer. The third one is the output folder path. The output result will be in this folder. The example is shown below.

The example is shown below:
### Detection

``` shell
python run_model.py detection ./data/Public_Private_Testing_Dataset_Only_for_detection ./runs --weight weights/best.tflite
```


### Segmentation

``` shell
python run_model.py segmentation ./data/Public_Private_Testing_Dataset ./runs --weight weights/best.tflite
```



You will get the results in `runs` folder:
```
.
└──runs
   ├──object_detections
   │  └──submission.csv
   └──segmentation
      ├──0001.png
      ├──...
      └──0007.png
```

