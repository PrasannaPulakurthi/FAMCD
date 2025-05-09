# Digits Datasets
python main.py --src=mnist --tgt=usps
python main.py --src=usps --tgt=mnist
python main.py --src=svhn --tgt=mnist

# Traffic Sign Datasets
python main.py --src=synsig --tgt=gtsrb --image_size=40
python main.py --src=gtsrb --tgt=synsig --image_size=40
