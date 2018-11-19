# LSTM for driving behaviour classification

This is the code repository for our paper, [Driving Behavior Classification Based on Sensor Data Fusion Using LSTM Recurrent Neural Networks](https://ieeexplore.ieee.org/document/8317835).  

This repo contains the data and tensorflow code for training and testing the LSTM model we used in the paper. To run this code, it is necessary to have installed:  
- Python2.7
- Tensorflow == 0.12
- Numpy
- Sklearn

### Citation
If you find this work useful in your research, please consider citing:

    @inproceedings{saleh2017driving,
      title={Driving behavior classification based on sensor data fusion using lstm recurrent neural networks},
      author={Saleh, Khaled and Hossny, Mohammed and Nahavandi, Saeid},
      booktitle={2017 IEEE 20th International Conference on Intelligent Transportation Systems (ITSC)},
      pages={1--6},
      year={2017},
      organization={IEEE}
    }
## Usage 
1. Clone repo via `git clone https://github.com/KhaledSaleh/driving_behaviour_classification.git && cd driving_behaviour_classification.git`.
2. Run `python main.py --help` to check the available command line args.
3. Run `python main.py` to train the model on the full dataset.
