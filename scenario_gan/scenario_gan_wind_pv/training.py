import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as grad
import numpy as np
from .gan_model import Generator
from .gan_model import Discriminator
from .gan_model import weights_init
import matplotlib.pyplot as plt

# how_many days to train
how_many_train = 300  # 拿前300天的数据训练用于训练，后面保留60天测试
def train_gan(dataset, params, gan_type, dis_train_steps, gen_train_steps, SAVE_DIR, dtype):
    train_params = params['train_params']    
    epochs = train_params['epochs']
    minibatchsize = train_params['minibatchsize']
    lrorigin = train_params['wgan_lr']  

    ## setup generator and discriminator
    gen = Generator(params['gen_params']).type(dtype)
    dis = Discriminator(params['dis_params']).type(dtype)

    ## initial weights and bias
    gen.apply(weights_init)
    dis.apply(weights_init)

    dis_optimizer = optim.RMSprop(dis.parameters(), lr=lrorigin)
    gen_optimizer = optim.RMSprop(gen.parameters(), lr=lrorigin)

    print('start training :')
    condition=torch.zeros([how_many_train, 1, 1, 288]).type(dtype)
    reals=torch.zeros([how_many_train, 1, 1, 288]).type(dtype)
    for day in range(how_many_train):
        condition[day,:]=dataset[day, :] # 根据今天的
        reals[day,:]= dataset[day+1, :] # 目标明天的
    
    condition=condition.reshape(how_many_train, 1, 12, 24)
    reals=reals.reshape(how_many_train, 1, 12, 24) 

    losses = np.zeros([epochs, 1])

    for epoch in range(epochs):  
        if (epoch+1) % (epochs/100) == 0:
            print('% :', int(100*((epoch+1)/epochs)), 'epoch:', epoch)  

        ## shuffle
        random_idx = torch.randperm(int(condition.shape[0])).type(torch.cuda.LongTensor)
        for iteration in range(int(random_idx.shape[0]/minibatchsize)):
            ## create mini-batches
                batch_idx = random_idx[iteration*minibatchsize:(iteration+1)*minibatchsize]
                for i in range(dis_train_steps):
                    dis.zero_grad()

                    real_input=torch.cat([condition[batch_idx,:], reals[batch_idx,:]], 2)
                    dis_real_out = dis(real_input)   
                    noise_data=torch.randn((minibatchsize, 1, 12, 24)).type(dtype)
                    noise_input=torch.cat([condition[batch_idx,:], noise_data], 2)
                    gen_out = gen(noise_input.type(dtype)).detach() 

                    gen_out_combinate=torch.cat([condition[batch_idx,:], gen_out], 2) # [4, 1, 12, 24]
                    dis_fake_out = dis(gen_out_combinate)

                    dis_real_loss = torch.mean(dis_real_out)
                    dis_fake_loss = torch.mean(dis_fake_out)
               
                    # gradient penalty
                    alpha = torch.rand(minibatchsize,1,24,24)
                    alpha = alpha.cuda()
                    x_hat = alpha *real_input + (1 - alpha) *gen_out_combinate
                    x_hat.requires_grad = True
                    pred_hat = dis(x_hat)
                    gradients = grad.grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                    create_graph=True, retain_graph=True, only_inputs=True)[0] 
                    gradient_penalty = 1*((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()#x.norm(级数p=2,维度dim=1)
                    gradient_penalty=torch.Tensor(gradient_penalty)

                    #损失更新
                    dis_loss = -(dis_real_loss - dis_fake_loss) + gradient_penalty            
                    dis_loss.backward()
                    dis_optimizer.step()                    
                
                losses[epoch, 0] = dis_loss # 监测一个iter，不是epoch

            ## train generator (i-times per epoch):
                for i in range(gen_train_steps):            
                    gen.zero_grad()
                    gen_noise_data=torch.randn(minibatchsize, 1, 12, 24).type(dtype)
                    gen_in=torch.cat([condition[batch_idx,:],gen_noise_data], 2)
                    gen_out = gen(gen_in)
                    gen_out_combinate = torch.cat([condition[batch_idx,:], gen_out], 2)
                    dis_fake_out = dis(gen_out_combinate)

                    #损失更新
                    gen_loss = -torch.mean(dis_fake_out)
                    gen_loss.backward()
                    gen_optimizer.step()

        if (epoch+1) % 50 == 0 or epoch+1 == epochs:
            torch.save(gen.state_dict(), f'D:\\scenario_gan\\aaasave\\generator_{epoch}.pth')
            plt.plot(losses, label="Discriminator Loss")
            plt.ylim(0, losses[epoch]+losses[epoch-1]+losses[epoch-2])
            plt.savefig(f'D:\\scenario_gan\\aaasave\\loss_epoch{epoch+1}.png')
            plt.close()

    print('done!')