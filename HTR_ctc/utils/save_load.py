'''avoid saving and loading directly to gpu'''
from collections import OrderedDict
import numpy as np
import torch
from torch.nn.parameter import Parameter
#import torch.nn as nn


def my_torch_save(model, filename):
    torch.save(model.state_dict(), filename)


def my_torch_load_old(model, filename, use_list=None):

    model_parameters = torch.load(filename)
    own_state = model.state_dict()

    for name in model_parameters.keys():
        if use_list is not None:
            if name not in use_list:
                continue
        if name in own_state:
            #print name
            param = model_parameters[name]
            if isinstance(param, Parameter):
                param = param.data

            if own_state[name].shape[:] != param.shape[:]:
                print(name)
                continue
            own_state[name].copy_(param)
        else:
            print(name)

def my_torch_load(model, filename):

    load_parameters = torch.load(filename)
    model.load_state_dict(load_parameters)
    # # When its trained on GPU modules will have 'module' in their name
    # is_module_named = np.any([k.startswith('module') for k in load_parameters.keys() if k])
    # if is_module_named:
    #     # Remove module. prefix from keys
    #     new_state_dict = OrderedDict()
    #     for k, v in load_parameters.items():
    #         if k.startswith('module.'):
    #             k = k.replace('module.', '')
    #         new_state_dict[k] = v
    #     state_dict = new_state_dict
    #
    # # Load the weights into the model
    # try:
    #     model.load_state_dict(state_dict)
    # except Exception as exp:
    #     print(exp)


