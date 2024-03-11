# Transformer fusion Pipeline

import sys, os, pickle, yaml, torch, logging, time, torchvision, copy, random
sys.path.append(os.path.join(os.path.dirname(__file__), "otfusion"))
from otfusion.hf_vit_fusion import hf_vit_fusion
from otfusion.utils import get_activations, accumulate_nested_dicts, divide_nested_dicts, multi_model_vanilla, model_to_dict, vanilla_fusion_old, model_eq_size_check
sys.path.append(os.path.join(os.path.dirname(__file__), "vit"))
from vit import vit_helper
import numpy as np

from datasets import config as ds_config

def main(exp = None, exp_mod = None, log_file = None):
    """
    ## Description
    The main function implements a full otfusion, evaluation and finetuning pipeline. The function implements the following steps:
    1. Initialize logger
    2. Read YAML file config.
    3. Modify config (if exp_mod is not None)
    4. Load models
    5. Compute activations
    6. Perform OTFusion
    7. Perform vanilla-fusion
    8. Evaluate one-shot accuracy (pre-finetuning)
    9. Finetuning
    10. Evaluate post-finetuning performance
    ------
    ## Parameters
    `exp`       experiment name string (i.e. `fuse_enc_dec_gen_N1_sort.yaml`)\\
    `exp_mod`   either dictionary containing modifications to the experiment config, or the flag 'is_sweep' indicating a wandb sweep
                Note:    dictionary must have the same structure as the experiment
                Example:    The following exp_mod dict would change the num_samples to 50
                            and the switch off the generator fusion:
                            `exp_mod = {'fusion': {'acts': {'num_samples': 50}}, 'fuse_gen': False}`
    `log_file`  relative or full file path + name of the logfile where the function should write to.
                Note:   Each function call of the main function should have a unique log_file name
                        if they are run in parallel, else the log files can get corrupted.
                Example: `reports/14_03_2023_regression/1.log`
    """
    # Default experiment
    EXPERIMENT_CFG_FILE = 'experiments/fuse_hf_vit_cifar10.yaml'
    LOGGING_LEVEL       = logging.INFO

    # Initialize logger
    if len(sys.argv) > 1:
        if (any('--debug' in string for string in sys.argv)):
            LOGGING_LEVEL = logging.DEBUG
    if log_file != None:
        log = logging.getLogger('{0}_main'.format(log_file))
        fileHandler = logging.FileHandler(log_file, mode='a')
        log.addHandler(fileHandler)
    else:
        log = logging.getLogger('main')
    logging.basicConfig(level=LOGGING_LEVEL)

    # Load Experiment Configuration
    args = load_args(log = log, EXPERIMENT_CFG_FILE = EXPERIMENT_CFG_FILE, exp = exp)


    # Print experiment configuration to commandline
    log_args(log = log, args = args, exp_mod = exp_mod)

    device = torch.device('cpu')

    # Set all seeds
    SEED = args['fusion']['acts']['seed']
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Set a different directory for datasets if specified in the environment variables
    # Defaults to huggingface cache otherwise
    ds_path = os.environ.get("HF_DATASET_PATH")
    if ds_path is not None:
        ds_config.DOWNLOADED_DATASETS_PATH = ds_path
        ds_config.HF_DATASETS_CACHE = ds_path

    # Load Models
    log.info(" ------- Loading Models -------\n")
    weights = {}
    models = []
    for i in range(args['fusion']['num_models']):
        weights['model_{0}'.format(i)], model = load_weights_and_model(args, key = "name_{0}".format(i))
        models.append(model)

    # check wheter models are of same size --> if models are of different size, vanilla fusion cannot be applied
    args['fusion']['heterogeneous'] = not model_eq_size_check(models, log)
    if args['fusion']['heterogeneous']:
        log.info(" Models have different sizes")
    else:
        log.info(" Models are of equal size")


    # compute activations
    log.info(" ------- Computing Activations -------\n")
    dataloader = get_dataloader(args, device)
    start_time = time.perf_counter()
    acts = get_activations(args = args, models = models, dataloader = dataloader, LOGGING_LEVEL = LOGGING_LEVEL, device = device, log_file = log_file)
    end_time = time.perf_counter()
    log.info(' Time for computing activations: {0:.4f} s'.format(end_time - start_time))

    # otfusion
    log.info(" ------- Performing OTFusion -------\n")
    start_time = time.perf_counter()
    # claculate alpha (used for valley plot or multi model fusion)
    # Note: Alpha decides how much of model 0 and how much of model 1 should be kept
    alpha = 1 / args['fusion']['num_models']
    anker_weights = weights['model_1']
    anker_acts    = acts['model_1']
    w_fused_list  = []
    for i in range(args['fusion']['num_models']-1):
        index = i
        if index > 0:
            log.info(' -------')
            index += 1 # model_1 is always the anker --> model 1 must not be fused with model 1
        # separate fusion of anker_model + model_i
        log.info(' Fusing anker model (model_1) with model_{0}'.format(index))
        w_fused_list.append(do_otfusion(args = args, weights = {'model_1': anker_weights, 'model_0': weights['model_{0}'.format(index)]},
                                        acts = {'model_1': anker_acts, 'model_0': acts['model_{0}'.format(index)]}, alpha = alpha, device = device, LOGGING_LEVEL = LOGGING_LEVEL, log_file = log_file))
    # accumulate weights
    for i in range(args['fusion']['num_models']-1):
        if i == 0:
            w_fused_acc = w_fused_list[0]
        else:
            w_fused_acc = accumulate_nested_dicts(w_fused_acc, w_fused_list[i])
    # divide by num_models - 1
    w_fused = divide_nested_dicts(w_fused_acc, args['fusion']['num_models']-1)
    end_time = time.perf_counter()
    log.info(' Time for OTFusion: {0:.4f} s'.format(end_time - start_time))
    # w_fused['config'] = weights['model_1']['config']
    model_otfused = get_model(args, w_fused)

    # vanilla fusion
    log.info(" ------- Performing Vanilla Fusion -------\n")
    if not args['fusion']['heterogeneous']:
        start_time = time.perf_counter()
        model_vanilla_fused = do_vanilla_fusion(args, weights, models[0], models[1])
        end_time = time.perf_counter()
        log.info(' Time for vanilla fusion: {0:.4f} s'.format(end_time - start_time))
    else:
        log.info(" Vanilla fusion not possible for models with different sizes")

    # Delete weights and acts from memory
    del weights
    del acts
    torch.cuda.empty_cache()

    # Evaluation
    log.info(" ------- Evaluating Models -------")
    test_dataloader = get_test_dataloader(args, device)
    if args.get("regression", {}).get("only_eval_ot", False) == False:
        for i in range(args['fusion']['num_models']):
            test_accuracy = get_test_acc(args, models[i], test_dataloader, device)
            log.info(" Model {0} Accuracy: {1}".format(i, test_accuracy))

    test_accuracy = get_test_acc(args, model_otfused, test_dataloader, device)
    log.info(" OTfusion Accuracy: {0}".format(test_accuracy))
    
    if args.get("regression", {}).get("only_eval_ot", False) == False and not args['fusion']['heterogeneous']:
        test_accuracy = get_test_acc(args, model_vanilla_fused, test_dataloader, device)
        log.info(" Vanilla Fusion Accuracy: {0}".format(test_accuracy))        


# Loading Arguments from experiment file
def load_args(log, EXPERIMENT_CFG_FILE, exp = None):
    """
    There are three ways in which an experiment can be defined. Below is a list ordered by priority (only experiment with highest priority is carried out)
    1. Main function input parameter 'exp'
    2. Command line specified
    3. Default experiment
    """
    if exp == None:
        if len(sys.argv) > 1:
            indices = [sys.argv.index(string) for string in sys.argv if '.yaml' in string]
            if (len(indices) > 0):
                assert(len(indices) == 1) # cannot specify multiple yaml files!
                EXPERIMENT_CFG_FILE = 'experiments/{0}'.format(sys.argv[indices[0]])
                log.info(" Running command line specified experiment: {0}".format(EXPERIMENT_CFG_FILE))
            else:
                log.info(" Using predefined experiment: {0}".format(EXPERIMENT_CFG_FILE))
        else:
            log.info(" Using predefined experiment: {0}".format(EXPERIMENT_CFG_FILE))
    else:
        EXPERIMENT_CFG_FILE = 'experiments/{0}'.format(exp)
        log.info(" Using experiment file defined by main function input parameter: {0}".format(EXPERIMENT_CFG_FILE))
    log.info(" ------- Reading Experiment Configuration -------\n")
    cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), EXPERIMENT_CFG_FILE.split("/")[0], EXPERIMENT_CFG_FILE.split("/")[1])
    with open(cfg_file, 'r') as f:
        args = yaml.safe_load(f)
    return args    
    

def log_args(log, args, exp_mod):
    log.debug('\n{0}'.format(yaml.dump(exp_mod, indent=4)))
    log.info('\n{0}'.format(yaml.dump(args, indent=4)))

#      _             _     _ _            _                    ____                  _  __ _
#     / \   _ __ ___| |__ (_) |_ ___  ___| |_ _   _ _ __ ___  / ___| _ __   ___  ___(_)/ _(_) ___
#    / _ \ | '__/ __| '_ \| | __/ _ \/ __| __| | | | '__/ _ \ \___ \| '_ \ / _ \/ __| | |_| |/ __|
#   / ___ \| | | (__| | | | | ||  __/ (__| |_| |_| | | |  __/  ___) | |_) |  __/ (__| |  _| | (__
#  /_/   \_\_|  \___|_| |_|_|\__\___|\___|\__|\__,_|_|  \___| |____/| .__/ \___|\___|_|_| |_|\___|
#                                                                   |_|

def load_weights_and_model(args, key):
    """
    ## Description
    Loads either model or model weights from memory and returns both the model and the
    corresponding nested weights dictionary containing all the weights.
    ------
    ## Parameters
    `args` Dictionary from YAML-based configuration file\\
    `key` Model key to retrive the model that should be loaded from the experiment dictionary (usual values `name_0` and `name_1`)
    ------
    ## Outputs
    `weights` Nested dictionary containing only the weights of the model\\
    `model` Pytorch model object
    """
    if args['model']['type'] == 'hf_vit':
        model = vit_helper.get_model('{0}'.format(args['model'][key]))
        weights = model_to_dict(model)
    else:
        raise NotImplementedError
    return weights, model

def get_model(args, weights):
    """
    ## Description
    Transforms the nested weights dictionary into a pytorch model object
    ------
    ## Parameters
    `args` Dictionary from YAML-based configuration file\\
    `weights` Nested dictionary containing only the weights of the model
    ------
    ## Outputs
    `model` Pytorch model object
    """
    if args['model']['type'] == 'hf_vit':
        model = vit_helper.get_model('{0}'.format(args['model']['name_1']))
        for name, _ in model.named_parameters():
            words = name.split('.')
            temp_model = model
            temp_dict = weights
            # if words[-1] == "weight":
            for w in words[:-1]:
                # Navigating the tree
                temp_model = getattr(temp_model, w)
                temp_dict = temp_dict[w]
            setattr(temp_model, words[-1], torch.nn.parameter.Parameter(temp_dict[words[-1]]))
    else:
        raise NotImplementedError
    return model

def get_dataloader(args, device):
    """
    ## Description
    Loads the dataloader from memory.
    Exceptions: For hugginface models not a dataloader is loaded but instead the raw dataset!
    The dataloader generated by this function will be used in the forward_pass() function in the get_activation() function.
    NOTE:   Two get_dataloader functions exist (get_dataloader(), get_test_dataloader()) to allow for different batch sizes
            during testing and in the get_activation() function.
    ------
    ## Parameters
    `args` Dictionary from YAML-based configuration file\\
    `device` Pytorch device object
    ------
    ## Outputs
    `dataloader` dataloader object
    """
    if args['model']['type'] == 'hf_vit':
        val_ds, test_ds = vit_helper.load_dataset_vit(args['model']['dataset'], args['fusion']['acts']['seed'])
        # Create a Dataloader with torch
        dataloader = torch.utils.data.DataLoader(dataset=val_ds,
                                                collate_fn=vit_helper.collate_fn,
                                                batch_size=1,
                                                shuffle=False)
    else:
        raise NotImplementedError
    return dataloader

def get_test_dataloader(args, device):
    """
    ## Description
    Loads the dataloader from memory.
    Exceptions: For hugginface models not a dataloader is loaded but instead the raw dataset!
    The dataloader generated by this function will be used for testing the base models, the otfused model and the vanilla fused model.
    NOTE:   Two get_dataloader functions exist (get_dataloader(), get_test_dataloader()) to allow for different batch sizes
            during testing and in the get_activation() function.
    ------
    ## Parameters
    `args` Dictionary from YAML-based configuration file\\
    `device` Pytorch device object
    ------
    ## Outputs
    `test_dataloader` dataloader object
    """
    if args['model']['type'] == 'hf_vit':
        _, test_dataloader = vit_helper.load_dataset_vit(args['model']['dataset'])
    else:
        raise NotImplementedError
    return test_dataloader

def do_otfusion(args, weights, acts, alpha, device, LOGGING_LEVEL, log_file):
    """
    ## Description
    Perform otfusion of two
    models.
    ------
    ## Parameters
    `args` Dictionary from YAML-based configuration file\\
    `weights` Weight dictionary containing the weights of both models (typical structure: `{model_0: {...}, model_1: {...}}`\\
    `acts` Activations dictionary containing all activations of both models (typical structure: `{model_0: {...}, model_1: {...}}`\\
    `alpha` Weighting parameter for anker model\\
    `device` Pytorch device object\\
    `LOGGING_LEVEL` Logging level\\
    `log_file` Path to logfile
    ------
    ## Outputs
    `w_fused` Nested dictionary containing only the weights of the fused model
    """
    if args['model']['type'] == 'hf_vit':
        w_fused = hf_vit_fusion(args = args, weights = weights, acts = acts, alpha = alpha, device = device, LOGGING_LEVEL = LOGGING_LEVEL, log_file = log_file)
    else:
        raise NotImplementedError
    return w_fused

def do_vanilla_fusion(args, weights, model_0, model_1):
    """
    ## Description
    Perform vanilla fusion of two
    models.
    ------
    ## Parameters
    `args` Dictionary from YAML-based configuration file\\
    `weights` Weight dictionary containing the weights of both models (typical structure: `{model_0: {...}, model_1: {...}}`\\
    `model_0` Pytorch model object of model 0\\
    `model_1` Pytorch model object of model 1
    ------
    ## Outputs
    `model_vanilla_fused` Pytorch object of vanilla-fused model
    """
    if args['model']['type'] == 'hf_vit':
        if args['fusion']['num_models'] > 2:
            w_vf_fused = multi_model_vanilla(args, weights)
            model_vanilla_fused = get_model(args, w_vf_fused)
        else:
            model_vanilla_fused = vit_helper.get_model('{0}'.format(args['model']['name_0']))
            model_vanilla_fused = vanilla_fusion_old(model_0, model_1, model_vanilla_fused)
    else:
        raise NotImplementedError
    return model_vanilla_fused

def get_test_acc(args, model, dataloader, device):
    """
    ## Description
    Tests model and returns
    accuracy over the test set.
    ------
    ## Parameters
    `args` Dictionary from YAML-based configuration file\\
    `model` Pytorch model object\\
    `dataloader` Dataloader objet\\
    `device` Pytorch device object
    ------
    ## Outputs
    `acc` Accuracy
    """
    if args['model']['type'] == 'hf_vit':
        acc = vit_helper.evaluate_vit(model, dataloader)
    else:
        raise NotImplementedError
    return acc


if __name__ == '__main__':
    main()
