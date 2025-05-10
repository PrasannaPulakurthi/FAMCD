# Digits Datasets
python main.py --src=mnist --tgt=usps
python main.py --src=usps --tgt=mnist
python main.py --src=svhn --tgt=mnist

# Traffic Signs Datasets
python main.py --src=synsig --tgt=gtsrb --lambda_fa=0.1

# Syn2Real Dataset
python main.py --src=syn --tgt=real --image_size=224
