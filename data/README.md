## GTSRB

### Option 1: Download the Dataset from the [Official Website](https://benchmark.ini.rub.de/). 

To prepare the data, run the code below after downloading and extracting the following files from [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html). 

```bash
python process_gtsrb.py
```

1. The official training data (use this to train your model):
 - Images and annotations (GTSRB_Final_Training_Images.zip)

2. The official test dataset (use this to test your model):
 - Images and annotations (without ground truth classes) (GTSRB_Final_Test_Images.zip)
 - Extended ground truth annotations (with classes) (GTSRB_Final_Test_GT.zip)

### Option 2: Download from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign). 

## SYNSET

### Download the Dataset from the Website. 

To prepare the data, run the code below after downloading and extracting the following files from [here](https://owncloud.fraunhofer.de/index.php/s/OLQ6E5BVN4pRGu8?path=%2FCyclesImagesOnly). 

```bash
process_synset.py
```
