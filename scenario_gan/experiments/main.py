import torch
import random
from config import get_params
import numpy as np

import sys
sys.path.insert(0, '../')
sys.path.append("D:\\scenario_gan")

from scenario_gan_wind_pv.training import train_gan

def main(data_key, DATA_DIR, SAVE_DIR):
    random_seed = 42   
    random.seed(random_seed)   
    torch.manual_seed(random_seed)
    if torch.cuda.is_available() == True:
        torch.cuda.set_device(0)
        dtype = torch.cuda.FloatTensor  #GPU
        torch.cuda.manual_seed_all(random_seed)
        print('running on gpu')
    else:
        dtype = torch.FloatTensor #CPU
        print('running on cpu')

    # setup parameter
    params = get_params(data_key)
    dataset = torch.load("D:\\scenario_gan\\dataset.pth")
    # start training
    train_gan(dataset, params, gan_type="wgan-GP", dis_train_steps=3, gen_train_steps=1, SAVE_DIR=SAVE_DIR, dtype=dtype)
        
if __name__ == '__main__':
    data_key = ['pv']
    DATA_DIR = "D:\\scenario_gan\\SONG"
    SAVE_DIR = "D:\\scenario_gan\\ZI"
            
    main(data_key, DATA_DIR, SAVE_DIR)
    
    
    
