
def _get_train_params(data_key):
    
    train_params = {'epochs' :2000,
              'minibatchsize' : 16, 
              'gan_lr' : 2*10e-4,
              'wgan_lr' : 2*10e-4}
       
    if data_key == 'pv':
        train_params['farms'] = 2
        train_params['hours'] = 96
    return train_params

def _get_dis_params(data_key):
    
    dis_params = {'input' : [1, 16, 32, 32],
                 'output' : [16, 32, 32, 64]}
    
    dis_params['kernel'] = [(3, 7), (3, 7), (3, 7), (3, 7)]
    dis_params['stride'] = [1, 1, 1, 1]
    dis_params['padding'] = [(1, 3), (1, 3), (1, 3), (1, 3)]
    
    """
    dis_params['kernel'] = [(5, 5), (5, 5), (5, 5), (5, 5)]
    dis_params['stride'] = [1, 1, 1, 1]
    dis_params['padding'] = [(2, 2), (2, 2), (2, 2), (2, 2)]
    """
    return dis_params

def _get_gen_params(data_key):

    gen_params = {'input' : [1, 64, 32, 16],
                 'output' : [64, 32, 16, 1]}

    gen_params['kernel'] = [(3, 7), (3, 7), (3, 7), (3, 7)]
    gen_params['stride'] = [1, 1, 1, (2, 1)]
    gen_params['padding'] = [(1, 3), (1, 3), (1, 3), (1, 3)]
    
    """
    gen_params['kernel'] = [(5, 5), (5, 5), (5, 5), (3, 3)]
    gen_params['stride'] = [1, 1, 1, (2, 1)]
    gen_params['padding'] = [(2, 2), (2, 2), (2, 2), (1, 1)]
    """
    return gen_params

def get_params(data_key):
    
    params = {'train_params' : _get_train_params(data_key),
             'dis_params' : _get_dis_params(data_key),
             'gen_params' : _get_gen_params(data_key)}
    
    return params
   