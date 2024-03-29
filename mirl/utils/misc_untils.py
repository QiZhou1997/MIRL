from collections import OrderedDict
import numpy as np
import torch

def get_scheduled_value(current, schedule):
    if type(schedule) in [float, int]:
        return schedule
    start, end, start_value, end_value = schedule
    ratio = (current - start) / (end - start) 
    ratio = max(0, min(1, ratio))
    value = (ratio * (end_value - start_value)) + start_value
    return value
    
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def to_list(
    x, 
    length=None, 
    pad_last_ietm=True,
    pad_item=None
):
    # None to []
    x = [] if x is None else x
    # item to [item] if type(item) is not a list
    x = x if type(x) is list else [x]
    # convert to a list with a given length. 
    # if pad_last_ietm is True, pad the list with the the last item.
    # else pad the list with pad_item 
    if length == 0:
        return []
    if length is not None:
        pad_item = x[-1] if pad_last_ietm else pad_item
        while len(x) < length:
            x.append(pad_item)
        assert len(x) == length
    return x

def split_item(item, valid_num):
    if type(item) is list or type(item) is np.ndarray:
        new_item = item[:valid_num]
        res_item = item[valid_num:]
    elif type(item) is dict or type(item) is OrderedDict:
        new_item = OrderedDict()
        res_item = OrderedDict()
        for key in item:
            new_item[key], res_item[key] = split_item(item[key], valid_num)
    else:
        raise NotImplementedError
    return new_item, res_item

def split_items(data, valid_num):
    data = [split_item(item, valid_num) for item in data]
    new_data, res_data = list(zip(*data))
    return new_data, res_data

def combine_item(item, cur_item):
    if type(item) is list:
        assert type(cur_item) is list
        new_item = item + cur_item
    elif type(item) is np.ndarray:
        assert type(cur_item) is np.ndarray
        new_item = np.concatenate([item, cur_item])
    elif type(item) is torch.Tensor:
        assert type(cur_item) is torch.Tensor
        new_item = torch.cat([item, cur_item])
    elif type(item) is dict or type(item) is OrderedDict:
        new_item = OrderedDict()
        for key in item:
            assert key in cur_item
            new_item[key] = combine_item(item[key], cur_item[key])
    else:
        raise NotImplementedError
    return new_item

def combine_items(data, cur_data):
    new_data = [combine_item(item, cur_item) for (item,cur_item) in zip(data,cur_data)]
    return tuple(new_data)


    