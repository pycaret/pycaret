from pyod.models.cof import COF
from pyod.models.sod import SOD
from pyod.models.sos import SOS


class COFPatched(COF):
    # This attribute was missing from pyod causing it to fail on clone
    @property
    def n_neighbors(self):
        return self.n_neighbors_

    @n_neighbors.setter
    def n_neighbors(self, value):
        self.n_neighbors_ = value


class SODPatched(SOD):
    # Those attributes were missing from pyod causing it to fail on clone
    @property
    def n_neighbors(self):
        return self.n_neighbors_

    @n_neighbors.setter
    def n_neighbors(self, value):
        self.n_neighbors_ = value

    @property
    def ref_set(self):
        return self.ref_set_

    @ref_set.setter
    def ref_set(self, value):
        self.ref_set_ = value

    @property
    def alpha(self):
        return self.alpha_

    @alpha.setter
    def alpha(self, value):
        self.alpha_ = value


class SOSPatched(SOS):
    def __init__(
        self, contamination=0.1, perplexity=4.5, metric="euclidean", eps=0.00001
    ):
        super().__init__(contamination, perplexity, metric, eps)
        # This is modified in pyod, which causes cloning to fail
        self.metric = metric
