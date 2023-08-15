# EfficentDet Model Training
This script is used to train and continue training EfficentDet models.

## Setup
1. Ensure Python 3.9 is installed: `sudo apt-get install python3.9 python3.9-distutils python3.9-dev`
2. Update alternatives: `sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1`
3. Install required Python packages: `pip install -r requirements.txt`
4. Install additional system packages: `sudo apt-get install python3-pip libportaudio2`

## Running the Training Script
1. Make sure your dataset paths are set correctly in `train.py`.
2. Set your batch size, epochs (add new epochs to original value, so if you trained 250 and want another 250 set it to 500)
3. Run the training with: `python train.py`.
