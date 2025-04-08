import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

from scenario_gan_wind_pv.gan_model import Generator
from experiments.config import get_params

params = get_params('pv')
data = np.genfromtxt('D:\\scenario_gan\\solar.txt', delimiter=',', dtype=None).astype(float)
dataset = torch.tensor(data, dtype=torch.float32) / 7 
dataset = dataset.reshape(365,288).cpu()

def interval_plot(day, scenior_gen_out, area_num=0):
    scenior_data= scenior_gen_out.cpu()
    quantile_values=np.zeros((1, 288, 8))
    for time in range(288):
        a=scenior_data[:, :, :, 0, time].reshape(-1)
        quantile_values[0, time, :]=np.percentile(a, [0, 10, 20, 30, 70, 80, 90, 100])

    y0=quantile_values[area_num, 0:288, 0]
    y10=quantile_values[area_num, 0:288, 1]
    y20=quantile_values[area_num, 0:288, 2]
    y30=quantile_values[area_num, 0:288, 3]
    y70=quantile_values[area_num, 0:288, 4]
    y80=quantile_values[area_num, 0:288, 5]
    y90=quantile_values[area_num, 0:288, 6]
    y100=quantile_values[area_num, 0:288, 7]

    x = np.arange(0, 288, 1)

    # 100%
    plt.fill_between(x, y0, y10, color='#008080', linewidth=0.2, label='100%', alpha=0.2)
    plt.fill_between(x, y90, y100, color='#008080', linewidth=0.2, alpha=0.2)
    # 90%
    plt.fill_between(x, y10, y20, color='#ffa500', linewidth=0.2, label='80%', alpha=0.4)
    plt.fill_between(x, y80, y90,color='#ffa500', linewidth=0.2, alpha=0.4)
    # 80%
    plt.fill_between(x, y20, y30, color='#ff7518', linewidth=0.2, label='60%', alpha=0.6)
    plt.fill_between(x, y70, y80, color='#ff7518', linewidth=0.2,  alpha=0.6)
    # 70%
    plt.fill_between(x, y30, y70, color='#ff69b4', label='40%', alpha=0.8)


for day in range(350,353):
    sec=50
    condition= dataset[day-1, :]
    condition=condition.reshape([1, 1, 12, 24])
    gen = Generator(params['gen_params']).type(torch.FloatTensor)
    gen_model=torch.load(r'D:\\scenario_gan\\aaasave\\generator_1999.pth',weights_only=True)
    gen.load_state_dict(gen_model)
    scenior_gen_out= torch.zeros(sec, 1, 1, 1, 288).type(torch.FloatTensor)
    for i in range(sec):
        noise_input=torch.randn(1, 1, 12, 24)
        gen_input=torch.cat([condition, noise_input], 2)
        gen_out = gen(gen_input.type(torch.FloatTensor)).reshape([1, 1, 1, 1, 288])
        scenior_gen_out[i, :]=gen_out.detach()
    #torch.save(scenior_gen_out, r'D:\\scenario_gan\\aaasave\\scenior_{}_{}.pth'.format(day, sec))

    mpl.rcParams['font.sans-serif']=['Times New Roman']
    plt.figure(figsize=(10, 6))
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 8}
    a = torch.zeros(1, 288)

    interval_plot(day,scenior_gen_out)

    for i in range(20-1): 
        a[0,:]=scenior_gen_out[i, 0, 0, 0, :].detach()
        plt.plot(a[0,:], linestyle='--', linewidth=0.8, alpha=0.7)
        plt.plot(scenior_gen_out[sec-1, 0, 0, 0, :], linestyle='--',linewidth=0.8, alpha=0.7)
        plt.legend(prop=font1, frameon=False)  
        plt.rcParams['savefig.dpi']=600   
        plt.tight_layout() 
    plt.show()

