
# Author: Moritz Imfeld <moimfeld@ethz.ch>
#
# Based on: https://github.com/sidak/otfusion/blob/eb73ad63905a8ac0b05e089e6ff8f7cbae803faf/ground_metric.py


import torch, logging

LOGGING_LEVEL = logging.INFO
log = logging.getLogger('ground_metric')
logging.basicConfig(level=LOGGING_LEVEL)

def isnan(x):
    return x != x

class GroundMetric:
    """
        Ground Metric object for Wasserstein computations:

    """
    def __init__(self, args, not_squared = False):
        self.ground_metric_type      = args['fusion']['gnd_metric']['type']
        self.ground_metric_normalize = args['fusion']['gnd_metric']['norm']
        self.reg                     = args['fusion']['gnd_metric']['reg']
        self.squared                 = args['fusion']['gnd_metric']['squared']
        self.mem_eff                 = args['fusion']['gnd_metric']['mem_eff']
        self.clip_max                = args['fusion']['gnd_metric']['clip_max']
        self.clip_min                = args['fusion']['gnd_metric']['clip_min']
        self.clip_gm                 = args['fusion']['gnd_metric']['clip']
        self.dist_normalize          = args['fusion']['acts']['norm']
        self.act_num_samples         = args['fusion']['acts']['num_samples']
        self.geom_ensemble_type      = args['fusion']['type']
        self.normalize_wts           = args['fusion']['wts']['norm']
        self.percent_clipped         = None
        self.args                    = args


    def _clip(self, ground_metric_matrix):

        percent_clipped = (float((ground_metric_matrix >= self.reg * self.clip_max).long().sum().data) \
                           / ground_metric_matrix.numel()) * 100
        print("percent_clipped is (assumes clip_min = 0) ", percent_clipped)
        self.percent_clipped = percent_clipped
        # will keep the M' = M/reg in range clip_min and clip_max
        ground_metric_matrix.clamp_(min=self.reg * self.clip_min,
                                             max=self.reg * self.clip_max)
        return ground_metric_matrix

    def _normalize(self, ground_metric_matrix):

        if self.ground_metric_normalize == "log":
            ground_metric_matrix = torch.log1p(ground_metric_matrix)
        elif self.ground_metric_normalize == "max":
            log.debug( "Normalizing ground metric by its max ({0:.4f})".format(ground_metric_matrix.max()))
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.max()
        elif self.ground_metric_normalize == "median":
            log.debug(" Normalizing ground metric by its median ({0:.4f})".format(ground_metric_matrix.median()))
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.median()
        elif self.ground_metric_normalize == "mean":
            log.debug(" Normalizing ground metric by its mean ({0:.4f})".format(ground_metric_matrix.mean()))
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.mean()
        elif self.ground_metric_normalize == "none":
            return ground_metric_matrix
        else:
            raise NotImplementedError

        return ground_metric_matrix

    def _sanity_check(self, ground_metric_matrix):
        assert not (ground_metric_matrix < 0).any()
        assert not (isnan(ground_metric_matrix).any())

    def _cost_matrix_xy(self, x, y, p=2, squared = True):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
        if not squared:
            c = c ** (1/2)
        if self.dist_normalize:
            assert NotImplementedError
        return c.detach().numpy()


    def _pairwise_distances(self, x, y=None, squared=True):
        dist = torch.cdist(x, y, p=2)
        if squared:
            dist = torch.pow(dist, 2)
        return dist.cpu().detach().numpy()

    def _get_euclidean(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            matrix = torch.norm(
                coordinates.view(coordinates.shape[0], 1, coordinates.shape[1]) \
                - coordinates, p=2, dim=2
            )
        else:
            if self.mem_eff:
                matrix = self._pairwise_distances(coordinates, other_coordinates, squared=self.squared)
            else:
                matrix = self._cost_matrix_xy(coordinates, other_coordinates, squared = self.squared)
        return matrix

    def _normed_vecs(self, vecs, eps=1e-9):
        norms = torch.norm(vecs, dim=-1, keepdim=True)
        log.debug("stats of vecs are: mean {}, min {}, max {}, std {}".format(
            norms.mean(), norms.min(), norms.max(), norms.std()
        ))
        return vecs / (norms + eps)

    def _get_cosine(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            matrix = coordinates / torch.norm(coordinates, dim=1, keepdim=True)
            matrix = 1 - matrix @ matrix.t()
        else:
            matrix = 1 - torch.div(
                coordinates @ other_coordinates.t(),
                torch.norm(coordinates, dim=1).view(-1, 1) @ torch.norm(other_coordinates, dim=1).view(1, -1)
            )
        return matrix.clamp_(min=0)

    def _get_angular(self, coordinates, other_coordinates=None):
        pass

    def get_metric(self, coordinates, other_coordinates=None):
        get_metric_map = {
            'euclidean': self._get_euclidean,
            'cosine': self._get_cosine,
            'angular': self._get_angular,
        }
        return get_metric_map[self.ground_metric_type](coordinates, other_coordinates)

    def process(self, coordinates, other_coordinates=None):
        if self.geom_ensemble_type == 'wts' and self.normalize_wts:
            log.debug("In weight mode: normalizing weights to unit norm")
            if self.args['model']['dataset'] == 'tinyimagenet':
                coordinates = coordinates / (torch.max(coordinates) + 1e-9)
                if other_coordinates is not None:
                    other_coordinates = other_coordinates / (torch.max(other_coordinates) + 1e-9)
            elif self.args['model']['dataset'] == 'wiki':
                coordinates = coordinates / (torch.max(torch.abs(coordinates)) + 1e-9)
                if other_coordinates is not None:
                    other_coordinates = other_coordinates / (torch.max(torch.abs(other_coordinates)) + 1e-9)
            else:    
                coordinates = self._normed_vecs(coordinates)
                if other_coordinates is not None:
                    other_coordinates = self._normed_vecs(other_coordinates)

        ground_metric_matrix = self.get_metric(coordinates, other_coordinates)

        self._sanity_check(ground_metric_matrix)

        ground_metric_matrix = self._normalize(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        if self.clip_gm:
            ground_metric_matrix = self._clip(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)


        return ground_metric_matrix
