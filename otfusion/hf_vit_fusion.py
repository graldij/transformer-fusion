# Top-level function for fusing two vision transformers using OTFusion
#
# Author: Moritz Imfeld <moimfeld@ethz.ch>

from otfusion_lib import ln_fusion, encoder_fusion, fc_fusion, resid_policy
import copy, logging, torch

#------------#
# VIT Fusion #
#------------#
def hf_vit_fusion(args: dict, weights: dict, acts: dict, alpha, device: torch.device, LOGGING_LEVEL, log_file = None):
    """
    ## Description
    Algorithm fuses the two transformers in a sequential manner (i.e. one element after another).
    A vision transformer is sturctured as follows:
    - Class Tokens, Embeddings
    - Encoders:
        - Layer Normalization (sublayer.norm.0)
        - Self-Attention Layer
        - Layer Normalization (sublayer.norm.1) (after this normalization, the residual is added)
        - Fully Connected Layer (after the fully connected layer, the residual is added)
    - Encoder norm (after the encoder chain)
    - Generator (head_1)
    ------
    ## Parameters
    `args` Dictionary from YAML-based configuration file\\
    `weights` Dictionary containing all weights of both transformer models that should be fused\\
    `acts` Dictionary containing all activations of both transfromer models that should be fused\\
    `alpha` Weighting parameter for anker model\\
    `device` torch.device()\\
    `LOGGING_LEVEL` logging level
    ------
    ## Outputs
    `w_fused` Dictionary containing fused weights
    """
    if log_file != None:
        log = logging.getLogger('{0}_otfusion'.format(log_file))
        fileHandler = logging.FileHandler(log_file, mode='a')
        log.addHandler(fileHandler)
    else:
        log = logging.getLogger('otfusion')
    log.setLevel(LOGGING_LEVEL)

    # init
    t_out              = None
    number_of_encoders = len(weights['model_0']['vit']['encoder']['layer'])
    w_fused            = {'vit': {'embeddings': {'patch_embeddings': {'projection': {}}},
                                  'encoder': {'layer': {}}}}

    # The otfusion_lib functions are designed to be transformer implementation agnostic. These functions operate on nested weight dictionaries without
    # direct knowledge of transformer module or layer names, as these details can vary. Instead, they rely on a predefined
    # dictionary of keys to access weights within the nested weight dictionary. To integrate a custom
    # transformer implementation with the otfusion_lib functions, on must define the corresponding values for
    # all keys in the keys variable such that the otfusion_lib functions can access the weights in the nested dictionary.
    keys = {}
    # Encoder keys
    keys['enc_ln0_keys']  = ['layernorm_before']
    keys['enc_ln1_keys']  = ['layernorm_after']
    keys['enc_sa_keys']   = ['attention']
    keys['enc_ff0_keys']  = ['intermediate', 'dense']
    keys['enc_ff1_keys']  = ['output', 'dense']

    # Attention keys
    keys['w_q']           = ['attention', 'query']
    keys['w_k']           = ['attention', 'key']
    keys['w_v']           = ['attention', 'value']
    keys['w_o']           = ['output', 'dense']

    # Fully connected
    keys['weights']       = ['weight']
    keys['bias']          = ['bias']

    # Layer norm
    keys['a']             = ['weight']
    keys['b']             = ['bias']

    if args['fusion']['fuse_src_embed']:
        # Fusing Class Token
        log.info(' Fusing class token')
        w_cls_token_0 = weights['model_0']['vit']['embeddings']['cls_token']
        w_cls_token_1 = weights['model_1']['vit']['embeddings']['cls_token']
        w_cls_token_0 = w_cls_token_0.squeeze(dim = 0)
        w_cls_token_1 = w_cls_token_1.squeeze(dim = 0)
        w_cls_token_fused, t_out = fc_fusion(args = args, keys = keys, t_in = None, w_0 = w_cls_token_0, w_1 = w_cls_token_1,
                                            act_0 = w_cls_token_0,
                                            act_1 = w_cls_token_1,alpha = alpha,  device = device, log = log,
                                            last_layer = False, is_embed = True, is_vit_embed = True)
        w_cls_token_fused = w_cls_token_fused['weight'].unsqueeze(dim = 0)
        w_fused['vit']['embeddings']['cls_token'] = w_cls_token_fused

        # Fusing Positional Embeddings
        log.info(' Fusing position embeddings')
        w_pos_embed_0 = copy.deepcopy(weights['model_0']['vit']['embeddings']['position_embeddings'])
        w_pos_embed_1 = copy.deepcopy(weights['model_1']['vit']['embeddings']['position_embeddings'])
        w_pos_embed_0 = w_pos_embed_0.squeeze(dim = 0)
        w_pos_embed_1 = w_pos_embed_1.squeeze(dim = 0)
        w_pos_embed_fused, t_out_pos = fc_fusion(args = args, keys = keys, t_in = None, w_0 = w_pos_embed_0, w_1 = w_pos_embed_1,
                                            act_0 = w_pos_embed_0,
                                            act_1 = w_pos_embed_1, alpha = alpha, device = device, log = log,
                                            last_layer = False, is_embed = True, is_vit_embed = True)
        w_pos_embed_fused = w_pos_embed_fused['weight'].unsqueeze(dim = 0)
        w_fused['vit']['embeddings']['position_embeddings'] = copy.deepcopy(w_pos_embed_fused)

        # Fuse Patch Embeddings
        log.info(' Fusing Patch Embeddings')
        # Idea -> align kernel-wise (each kernel has size [3, 4, 4])
        w_patch_embed_0 = copy.deepcopy(weights['model_0']['vit']['embeddings']['patch_embeddings']['projection'])
        w_patch_embed_1 = copy.deepcopy(weights['model_1']['vit']['embeddings']['patch_embeddings']['projection'])
        w_patch_embed_size = w_patch_embed_0['weight'].size()
        # transform from [384, 3, 4, 4] -> [384, 48]
        w_patch_embed_0['weight'] = w_patch_embed_0['weight'].view(-1, w_patch_embed_size[1] * w_patch_embed_size[2] * w_patch_embed_size[3])
        w_patch_embed_1['weight'] = w_patch_embed_1['weight'].view(-1, w_patch_embed_size[1] * w_patch_embed_size[2] * w_patch_embed_size[3])
        # fusion
        w_patch_embed_fused, t_out = fc_fusion(args = args, keys = keys, t_in = None, w_0 = w_patch_embed_0, w_1 = w_patch_embed_1,
                                              act_0 = acts['model_0']['vit']['embeddings']['patch_embeddings']['data'],
                                              act_1 = acts['model_1']['vit']['embeddings']['patch_embeddings']['data'],
                                              alpha = alpha, device = device, log = log, last_layer = False)
        # transform from [384, 48] -> [384, 3, 4, 4] (transform is inverse to pre-fusion transform)
        w_patch_embed_fused['weight'] = w_patch_embed_fused['weight'].view(-1, w_patch_embed_size[1], w_patch_embed_size[2], w_patch_embed_size[3]).detach()
        w_patch_embed_fused['bias']   = w_patch_embed_fused['bias'].detach()
        w_fused['vit']['embeddings']['patch_embeddings']['projection'] =  copy.deepcopy(w_patch_embed_fused)

        # combine transportation maps from positional encoding and patch embeddings
        t_out = resid_policy(policy = args.get('fusion').get('resid_policy'), t_in = t_out, t_resid = t_out_pos,
                             in_acts = acts['model_1']['vit']['embeddings']['patch_embeddings']['data'], resid_acts = w_pos_embed_1, log = log)
    else:
        log.info(' Copy Embeddings')
        w_fused['vit']['embeddings'] = copy.deepcopy(weights['model_1']['vit']['embeddings'])

    # Get the activations from the embedding output
    prev_out_acts = acts['model_1']['vit']['embeddings']['data']

    # fuse encoders
    for i in range(number_of_encoders):
        # init
        encoder_key = str(i)
        last_layer = (i == number_of_encoders-1) and not args['fusion']['fuse_gen']
        w_fused['vit']['encoder']['layer'][encoder_key], t_out = encoder_fusion(args = args, keys = keys, w_0 = weights['model_0']['vit']['encoder']['layer'][encoder_key],
                                                                                     w_1 = weights['model_1']['vit']['encoder']['layer'][encoder_key],
                                                                                     acts_0 = acts['model_0']['vit']['encoder']['layer'][encoder_key],
                                                                                     acts_1 = acts['model_1']['vit']['encoder']['layer'][encoder_key],
                                                                                     t_in = t_out, last_layer = last_layer, device = device, enc_key = encoder_key,
                                                                                     alpha = alpha, log = log, prev_out_acts = prev_out_acts)
        prev_out_acts = acts['model_1']['vit']['encoder']['layer'][encoder_key]['data']

    # Fuse Layer Normalization at the end of encoder chain
    log.info(' Fusing encoder output norm')
    w_fused['vit']['layernorm'], t_out = ln_fusion(args = args, keys = keys, t_in = t_out, w_0 = weights['model_0']['vit']['layernorm'],
                                                             w_1 = weights['model_1']['vit']['layernorm'],
                                                             alpha = alpha, device = device)

    # Fuse Classifier
    if args['fusion']['fuse_gen']:
        log.info(' Fusing classifier')
        w_fused['classifier'], t_out = fc_fusion(args = args, keys = keys, t_in = t_out, w_0 = weights['model_0']['classifier'],
                                             w_1 = weights['model_1']['classifier'], act_0 = acts['model_0']['classifier'],
                                             act_1 = acts['model_1']['classifier'],
                                             alpha = alpha, device = device, log = log, last_layer=True, is_vit_fc = True)
    else:
        log.info(' Skipping classifier fusion')
        w_fused['classifier'] = copy.deepcopy(weights['model_1']['classifier'])

    return w_fused
