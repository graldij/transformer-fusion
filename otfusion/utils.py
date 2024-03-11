# Utils
#
# Author: Moritz Imfeld <moimfeld@ethz.ch>

import numpy as np
import matplotlib.pyplot as plt
import os, copy, torch, logging, time
from functools import reduce
from operator import getitem

#   __  __       _        _        ____  _        _   _     _   _
#  |  \/  | __ _| |_ _ __(_)_  __ / ___|| |_ __ _| |_(_)___| |_(_) ___ ___
#  | |\/| |/ _` | __| '__| \ \/ / \___ \| __/ _` | __| / __| __| |/ __/ __|
#  | |  | | (_| | |_| |  | |>  <   ___) | || (_| | |_| \__ \ |_| | (__\__ \
#  |_|  |_|\__,_|\__|_|  |_/_/\_\ |____/ \__\__,_|\__|_|___/\__|_|\___|___/

def matrix_stats(matrix, name):
    """
    Returns a matrix in a string formatted way.
    """
    if torch.is_tensor(matrix):
        if matrix.is_cuda:
            matrix = matrix.cpu()
    numpy_matrix = np.asarray(matrix)
    shape  = numpy_matrix.shape
    sum    = numpy_matrix.sum()
    mean   = numpy_matrix.mean()
    median = np.median(numpy_matrix)
    max    = numpy_matrix.max()
    min    = numpy_matrix.min()
    histogram, bins = np.histogram(numpy_matrix, bins = 50)
    ret  = '\n{0} Matrix Stats:\n'.format(name)
    ret += 'Shape: {0}\n'.format(shape)
    ret += 'Sum: {0:.6f}, Mean: {1:.6f}, Median: {2:.6f}, Max: {3:.6f}, Min: {4:.6f}\n'.format(sum, mean, median, max, min)
    ret += 'Histogram: \n'
    ret += histogram_to_str(histogram = histogram, bins = bins)
    return ret

def histogram_to_str(histogram, bins):
    """
    Function turns histogram into a string that can be printed in the commandline.
    """
    # compute max length of bin_label and max number of digits of largerst histogram number
    max_len_label = 0
    for i in range(len(histogram)):
        length = len('{0:.5f}- {1:.5f}'.format(bins[i], bins[i+1]))
        if length > max_len_label:
            max_len_label = length
    # compute maximum number of 
    max_freq     = np.max(histogram)
    max_len_hist = len(str(max_freq))
    # get commandline width
    cmd_width = os.get_terminal_size().columns
    ret = ''
    for i in range(len(histogram)):
        ratio = histogram[i]/max_freq
        stars = int(round(ratio*(cmd_width - max_len_label - 2 - max_len_label - 10)))
        bin_label = '{0:.5f}- {1:.5f}'.format(bins[i], bins[i+1]).rjust(max_len_label)
        hist_cnt  = '{0}'.format(histogram[i]).rjust(max_len_hist)
        ret += '{0}: ({1}) {2}\n'.format(bin_label, hist_cnt, '*'*stars)
    return ret

def matrix_to_heatmap(matrix):
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.show()

#    ____      _        _        _   _            _   _
#   / ___| ___| |_     / \   ___| |_(_)_   ____ _| |_(_) ___  _ __
#  | |  _ / _ \ __|   / _ \ / __| __| \ \ / / _` | __| |/ _ \| '_ \
#  | |_| |  __/ |_   / ___ \ (__| |_| |\ V / (_| | |_| | (_) | | | |
#   \____|\___|\__| /_/   \_\___|\__|_| \_/ \__,_|\__|_|\___/|_| |_|

# Get activation constants
ONLY_CLS = torch.Tensor([    1., # cls_token
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')])
WINDOW_2 = torch.Tensor([    1., # cls_token
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'),           1.,           1., float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'),           1.,           1., float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')])
WINDOW_4 = torch.Tensor([    1., # cls_token
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'),           1.,           1.,           1.,           1., float('nan'), float('nan'),
                         float('nan'), float('nan'),           1.,           1.,           1.,           1., float('nan'), float('nan'),
                         float('nan'), float('nan'),           1.,           1.,           1.,           1., float('nan'), float('nan'),
                         float('nan'), float('nan'),           1.,           1.,           1.,           1., float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')])
WINDOW_6 = torch.Tensor([    1., # cls_token
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                         float('nan'),           1.,           1.,           1.,           1.,           1.,           1., float('nan'),
                         float('nan'),           1.,           1.,           1.,           1.,           1.,           1., float('nan'),
                         float('nan'),           1.,           1.,           1.,           1.,           1.,           1., float('nan'),
                         float('nan'),           1.,           1.,           1.,           1.,           1.,           1., float('nan'),
                         float('nan'),           1.,           1.,           1.,           1.,           1.,           1., float('nan'),
                         float('nan'),           1.,           1.,           1.,           1.,           1.,           1., float('nan'),
                         float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')])
FULL     = torch.Tensor([    1., # cls_token
                             1.,     1.,     1.,     1.,     1.,     1.,     1.,     1.,
                             1.,     1.,     1.,     1.,     1.,     1.,     1.,     1.,
                             1.,     1.,     1.,     1.,     1.,     1.,     1.,     1.,
                             1.,     1.,     1.,     1.,     1.,     1.,     1.,     1.,
                             1.,     1.,     1.,     1.,     1.,     1.,     1.,     1.,
                             1.,     1.,     1.,     1.,     1.,     1.,     1.,     1.,
                             1.,     1.,     1.,     1.,     1.,     1.,     1.,     1.,
                             1.,     1.,     1.,     1.,     1.,     1.,     1.,     1.])

def get_activations(args, models, dataloader, LOGGING_LEVEL, device, log_file = None):
    """
    1. Apply hooks to every layer of every model
    2. For each input perform forward pass for every model
    3. Apply transformations according to args (experiment configuration)
    4. Transform flat activation dict to nested dictionary structure
    """

    # init logger
    if log_file != None:
        log = logging.getLogger('{0}_get_activations'.format(log_file))
        fileHandler = logging.FileHandler(log_file, mode='a')
        log.addHandler(fileHandler)
    else:
        log = logging.getLogger('get_activations')
    log.setLevel(LOGGING_LEVEL)

    # init empty hook_dict
    hook_dict = {}

    # loop through the layers of the network and register the hooks
    start_time = time.perf_counter()
    apply_hooks(models = models, hook_dict = hook_dict)
    log.info(' Time for applying hooks: {0:.4f} s'.format(time.perf_counter() - start_time))

    # compute the activations for all models
    start_time = time.perf_counter()
    acts = get_raw_acts(args = args, hook_dict = hook_dict, dataloader = dataloader, models = models, log = log, device = device)
    log.info(' Time for get_raw_acts: {0:.4f} s'.format(time.perf_counter() - start_time))

    # apply transformations according to args (experiment configuration)
    start_time = time.perf_counter()
    acts = transform_acts(args = args, acts = acts, models = models, log = log, device = device)
    log.info(' Time for transform_acts: {0:.4f} s'.format(time.perf_counter() - start_time))

    # transform flat activation dict to nested dictionary structure
    for i in range(len(models)):
        modelkey = 'model_{0}'.format(i)
        acts[modelkey] = to_nest_dict(acts[modelkey], log)

    return acts

# Define a function that saves the activation of a layer
# Hook function was defined according to torch documentation
# Note: Value of activations[name] is overwritten if layer is
#       called multiple times. This behaviour is inteded
#       (and needed) because the decoder layers will be called
#       multiple times while generating the output sequence.
# Note: Some layers produce tuples as outputs. In these tuples
#       the first item is a torch.Tensor while the second item
#       is empty. The hook only write the tensor to the activations.
# Note: A hook will always capture the output of a given layer.
#       For example if we have an "attention" layer, the hook will
#       capture the output of that layer which would be
#       value*softmax(query*key.t())/dim(model)
def save_activation(name, activations):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activations[name] = output.detach()
        else:
            activations[name] = output[0].detach()
    return hook

# Apply hooks to every layer
def apply_hooks(models, hook_dict):
    for i, model in enumerate(models):
        modelkey = 'model_{0}'.format(i)
        hook_dict[modelkey] = {}
        for name, layer in model.named_modules():
            if 'dropout' not in name:
                layer.register_forward_hook(save_activation(name, hook_dict[modelkey]))

# Get raw activation
def get_raw_acts(args, hook_dict, dataloader, models, log, device):
    acts = {}
    num_itr = 0
    not_used_count = 0
    for itr, data in enumerate(dataloader):
        # initialize used as True
        used = True

        # If an acts_filter list is defined in the experiment and the 
        acts_filter_list = args.get('fusion', {}).get('acts', {}).get('acts_filter_list', None)
        if isinstance(acts_filter_list, list) and len(acts_filter_list) == itr:
            break
        # Forward Pass
        for model in models:
            used = used and forward_pass(args['model']['type'], args, model, data, itr, device)

        # only copy activations if return of forward_pass was true for every model
        # return of forward_pass returns true only if activation passed every configured filter (e.g. loss_thres)
        if used:
            for i, model in enumerate(models):
                modelkey = 'model_{0}'.format(i)
                # average activations if fusion->acts->avg_seq_items == True
                if args.get('fusion', {}).get('acts', {}).get('avg_seq_items', False) == True:
                    for key in hook_dict[modelkey]:
                        if len(hook_dict[modelkey][key].size()) == 3:
                            hook_dict[modelkey][key] = hook_dict[modelkey][key].mean(dim = 1)

                # Copy activations to dictionary if forward_pass returned true
                if num_itr == 0:
                    acts[modelkey] = {}
                    for key in hook_dict[modelkey]:
                        acts[modelkey][key] = [copy.deepcopy(hook_dict[modelkey][key])]
                else:
                    tmp_dict = {}
                    for key in hook_dict[modelkey]:
                        # NOTE: The squeeze(dim = 0) of dimension 0 is quite architecture specific to hf_bert_class, but there is no impact
                        #       of this transformation on other architectures (as of now)
                        #       The reson for this squeeze is that in the encoder layers the captured activations are of size [1, seq_length, 768]
                        #       The issue is that these activations cannot be concatenated at dim=0 --> so they are first sequeezed.
                        if len(hook_dict[modelkey][key].size()) == 3:
                            if hook_dict[modelkey][key].size()[1] != acts[modelkey][key][0].size()[1]:
                                hook_dict[modelkey][key] = hook_dict[modelkey][key].squeeze(dim = 0)
                                acts[modelkey][key][0] = acts[modelkey][key][0].squeeze(dim = 0)
                                # NOTEs: this fix is hardcoded.
                                if "encoder.layer.5.attention.output.dense"in key or "encoder.layer.5.output.dense" in key or "encoder.layer.5.intermediate.dense" in key:
                                    hook_dict[modelkey][key]=hook_dict[modelkey][key][0,:].unsqueeze(dim=0)
                        if len(acts[modelkey][key][0].size()) == 0:
                            acts[modelkey][key] = acts[modelkey][key].view(1)
                        if len(hook_dict[modelkey][key].size()) == 0:
                            hook_dict[modelkey][key] = hook_dict[modelkey][key].view(1)
                        acts[modelkey][key].append(copy.deepcopy(hook_dict[modelkey][key]))
            num_itr += 1
        else:
            not_used_count += 1
        # Break once number of samples is reached
        if num_itr >= args['fusion']['acts']['num_samples']:
            loss_thres = args.get('fusion', {}).get('acts', {}).get('loss_thres', False)
            if loss_thres != False:
                log.info(' Filtered {0} activations because loss was larger than {1} (loss_thes) in one of the models'.format(not_used_count, loss_thres, False))
            break

    # concatenate captured tensors
    for i, model in enumerate(models):
        modelkey = 'model_{0}'.format(i)
        for key in hook_dict[modelkey]:
            acts[modelkey][key] = torch.cat(acts[modelkey][key], dim = 0)

    return acts

# Activation transformations
def transform_acts(args, acts, models, log, device):
    # before applying mean and std deviation first flatten the sequence (or only pick a certain element in each sequence)
    # NOTE: The function flatten does not only flatten the sequence items but also applies specified filters (e.g. window_2, only_cls, window_4, ect.)
    for i in range(len(models)):
        modelkey = 'model_{0}'.format(i)
        for layer in acts[modelkey]:
            if args.get('fusion', {}).get('acts', {}).get('avg_seq_items', False) == False:
                acts[modelkey][layer] = flatten(args, acts[modelkey][layer], layer, log, device)

    # Apply specified transformation
    for i in range(len(models)):
        modelkey = 'model_{0}'.format(i)
        for layer in acts[modelkey]:
            if args['fusion']['acts']['std']:
                mean_acts = torch.mean(acts[modelkey][layer], dim=0)
                std_acts  = torch.std(acts[modelkey][layer], dim=0)
                # (acts[modelkey][layer] - mean_acts)/(std_acts + 1e-9)
                acts[modelkey][layer] = torch.div(torch.sub(acts[modelkey][layer], mean_acts), torch.add(std_acts, 1e-9))
            elif args['fusion']['acts']['center']:
                mean_acts = torch.mean(acts[modelkey][layer], dim=0)
                # acts[modelkey][layer] - mean_acts
                acts[modelkey][layer] = torch.sub(acts[modelkey][layer], mean_acts)
            elif args['fusion']['acts']['norm']:
                acts[modelkey][layer] = torch.nn.functional.normalize(acts[modelkey][layer], dim=0, p=2, eps=1e-12)

    # report transformation to commandline
    if args['fusion']['acts']['seq_pos'] == -1:
        seq_filter = args.get('fusion').get('acts').get('seq_filter', None)
        if args.get('fusion', {}).get('acts', {}).get('avg_seq_items', False) == True:
            log.info(" Averaged the activation sequence for every sample.".format(args['fusion']['acts']['seq_filter'][-1]))
            if seq_filter != None:
                log.warning(" seq_filter is not None but avg_seq_items is True; Which means that the seq_filter option is ignored.")
        else:
            if seq_filter == 'window_2' or seq_filter == 'window_4' or seq_filter == 'window_6' or seq_filter == 'window_8' and args['model']['type'] == 'hf_vit':
                log.info(" Flattening activations and only pick middle window ({0}x{0}) of activations (+ class token)".format(args['fusion']['acts']['seq_filter'][-1]))
            elif seq_filter == 'only_cls'  and args['model']['type'] == 'hf_vit':
                log.info(" Only taking class token as an activations".format(args['fusion']['acts']['seq_filter'][-1]))
            else:
                log.info(" Flattening activations (each sequence element becomes an activation)")
    else:
        log.info(" Only taking sequence element at position {0} as activation")
    if args['fusion']['acts']['std']:
        log.info(" Applying layerwise standartization of activations ((acts - mean_acts)/(std_acts + 1e-9))")
    elif args['fusion']['acts']['center']:
        log.info(" Applying layerwise centering of activations (acts - mean_acts)")
    elif args['fusion']['acts']['norm']:
        log.info(" Applying layerwise normalization of activations")
    else:
        log.info(" Using raw activations")

    return acts

# Flat dict to nested dict coversion
def to_nest_dict(flat_dict, log, sep='.'):
    # sort the keys lexographically
    flat_dict = dict(sorted(flat_dict.items()))
    # generate the nested dictionary
    nested_dict = {}
    for key, value in flat_dict.items():
        parts = key.split(sep)
        d = nested_dict
        for part in parts[:-1]:
            if part not in d:
                log.debug(" '{0}' was added to the nested dict".format(part))
                d[part] = {}
            elif not isinstance(d[part], dict):
                log.debug(" Value was found for key '{0}' (instead of dict). Creating a dict to make space for further data. Value can be retrieved from this dict with the key 'data'.".format(part))
                d[part] = {"data": d[part]}
            d = d[part]
        d[parts[-1]] = value
    return nested_dict


#    ____                      _
#   / ___| ___ _ __   ___ _ __(_) ___
#  | |  _ / _ \ '_ \ / _ \ '__| |/ __|
#  | |_| |  __/ | | |  __/ |  | | (__
#   \____|\___|_| |_|\___|_|  |_|\___|

def dict_get(key_list, dict):
    return reduce(getitem, key_list, dict)

def dict_write(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

def accumulate_nested_dicts(d1, d2):
    acc_dict = {}
    for k1, v1 in d1.items():
        v2 = d2[k1]
        if isinstance(v1, dict) and isinstance(v2, dict):
            acc_dict[k1] = accumulate_nested_dicts(v1, v2)
        else:
            sum = v1 + v2
            acc_dict[k1] = sum
    return acc_dict

def divide_nested_dicts(d1, dividend):
    acc_dict = {}
    for k1, v1 in d1.items():
        if isinstance(v1, dict):
            acc_dict[k1] = divide_nested_dicts(v1, dividend)
        else:
            sum = v1 / dividend
            acc_dict[k1] = sum
    return acc_dict

def detach_tensors_in_nested_dict(d1):
    acc_dict = {}
    type(d1)
    for k1, v1 in d1.items():
        if isinstance(v1, dict):
            acc_dict[k1] = detach_tensors_in_nested_dict(v1)
        elif isinstance(v1, torch.Tensor):
            acc_dict[k1] = v1.detach()
    return acc_dict

def multi_model_vanilla(args, weights):
    for i in range(args['fusion']['num_models']):
        if i == 0:
            weights['model_0'].pop('config', None)
            weights_acc = weights['model_0']
        else:
            weights['model_{0}'.format(i)].pop('config', None)
            weights_acc = accumulate_nested_dicts(weights_acc, weights['model_{0}'.format(i)])
    w_vf_fused = divide_nested_dicts(weights_acc, args['fusion']['num_models'])
    return w_vf_fused


def model_to_dict(model, is_vit=False):
    top_level = {}

    for name, param in model.named_parameters():
        words = name.split('.')
        active_dict = top_level
        for w in words[:-1]:
            if w not in active_dict:
                active_dict[w] = {}
            active_dict = active_dict[w]
        active_dict[words[-1]] = param
    

    top_level["config"] = model.config

    return top_level

def vanilla_fusion_old(model_1, model_2, model_fused):
    for (name_1, param_1), (name_2, param_2), (name_fused, param_fused) in zip(model_1.named_parameters(), model_2.named_parameters(), model_fused.named_parameters()):
        tensor_fused = param_1.data.add(param_2.data).div(2)
        param_fused.data = tensor_fused
    return model_fused

def model_eq_size_check(models, log):
    log.info("Starting model size equivalence check")
    all_named_parameters = []
    for model in models:
        all_named_parameters.append([param for name, param in model.named_parameters()])

    for param_idx in range(len(all_named_parameters[0])):
        for model_idx in range(len(models)-1):
            if all_named_parameters[model_idx][param_idx].size() != all_named_parameters[model_idx + 1][param_idx].size():
                log.info("Models sizes are not equivalent. First different layer: {0}".format([name for name, param in model.named_parameters()][param_idx]))
                return False

    return True


#      _             _     _ _            _                    ____                  _  __ _
#     / \   _ __ ___| |__ (_) |_ ___  ___| |_ _   _ _ __ ___  / ___| _ __   ___  ___(_)/ _(_) ___
#    / _ \ | '__/ __| '_ \| | __/ _ \/ __| __| | | | '__/ _ \ \___ \| '_ \ / _ \/ __| | |_| |/ __|
#   / ___ \| | | (__| | | | | ||  __/ (__| |_| |_| | | |  __/  ___) | |_) |  __/ (__| |  _| | (__
#  /_/   \_\_|  \___|_| |_|_|\__\___|\___|\__|\__,_|_|  \___| |____/| .__/ \___|\___|_|_| |_|\___|
#                                                                   |_|

# Forward pass (used in get_raw_acts())
def forward_pass(model_type, args, model, data, data_item_no, device):
    """
    Function performs a forward pass for the corresponding model architecture
    Function returns True or False depending on wheter the sample should be considered as a valid activation for the activation based fusion
    """
    loss_thres = args.get('fusion', {}).get('acts', {}).get('loss_thres', False)
    acts_filter_list = args.get('fusion', {}).get('acts', {}).get('acts_filter_list', None)
    # Forward pass (architecture dependent)
    if model_type == 'hf_vit':
        model.to(device)
        pred = model(data["pixel_values"].to(device))
        data = [None, data['labels']]
    else:
        raise NotImplementedError

    # determine if activation of this data should be considered
    if loss_thres != False:
        used = loss_thres_filter(model_type, pred, data[1], loss_thres)
    elif isinstance(acts_filter_list, list):
        used = acts_filter_list[data_item_no]
    else:
        used = True
    return used

# Loss filter
def loss_thres_filter(model_type, pred, target, loss_thres):
    if model_type == 'hf_vit':
        if isinstance(loss_thres, float):
            # Convert target to one-hot encoding
            target_one_hot = torch.nn.functional.one_hot(target, num_classes=10)
            # Compute loss
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(pred.logits, target_one_hot.float())
            if loss > loss_thres:
                used = False
            else:
                used = True
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return used

# Activation transformation (used in transform_acts())
def flatten(args, acts, layer_name, log, device):
    # Read sequence filter from args
    seq_filter = args.get('fusion').get('acts').get('seq_filter', None)

    # Architecture specific activation handling
    if args['model']['type'] == 'hf_vit':
        # Sequence filter
        # NOTE: THIS FUNCTION IS NOT GENERIC AND ONLY WORKS IF THE VIT operates on 65 patches
        # NOTE: ONLY WORKS FOR PATCHES OF SIZE 8 x 8
        if seq_filter != None:
            if acts.size()[1] == 65: # only perform window on activations with
                # Load filter
                if seq_filter == 'only_cls':
                    filter = ONLY_CLS
                elif seq_filter == 'window_2':
                    filter = WINDOW_2
                elif seq_filter == 'window_4':
                    filter = WINDOW_4
                elif seq_filter == 'window_6':
                    filter = WINDOW_6
                elif seq_filter == 'window_8':
                    filter = FULL
                else:
                    raise NotImplementedError
                
                # For the last layer consider only the cls token as an activation
                if "encoder.layer.6.attention.output.dense"in layer_name or "encoder.layer.6.output.dense" in layer_name or "encoder.layer.6.intermediate.dense" in layer_name:
                    filter = ONLY_CLS

                # Apply filter
                acts = torch.mul(acts, filter.view(1, -1, 1).to(device))

                # Remove float('nan') activations
                acts = acts.flatten(end_dim = 1)
                acts = acts[~torch.any(acts.isnan(),dim=1)]
            else:
                acts = acts.flatten(end_dim = 1)
        elif args.get('fusion').get('acts').get('seq_filter') == 'only_cls':
            if acts.size()[1] == 65:
                acts = acts[:,0,:]
            else:
                acts = acts.flatten(end_dim = 1)
        else:
            # Default flattening
            acts = acts.flatten(end_dim = 1)
    else:
        raise NotImplementedError
    return acts
