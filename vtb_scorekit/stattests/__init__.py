"""Available statistical tests.
For detailed information about statistical tests see module documentation.
"""
from .anderson_darling_stattest import _anderson_darling
from .chisquare_stattest import _chi_stat_test
from .cramer_von_mises_stattest import _cramer_von_mises
from .energy_distance import _energy_dist
from .epps_singleton_stattest import _epps_singleton
from .fisher_exact_stattest import _fisher_exact_stattest
from .g_stattest import _g_stat_test
from .hellinger_distance import _hellinger_distance
from .jensenshannon import _jensenshannon
from .kl_div import _kl_div
from .ks_stattest import _ks_stat_test
from .mann_whitney_urank_stattest import _mannwhitneyu_rank
from .mmd_stattest import _mmd_stattest
from .psi import _psi
from .t_test import _t_test2samp
from .tvd_stattest import _tvd_stattest
from .wasserstein_distance_norm import _wasserstein_distance_norm
from .z_stattest import _z_stat_test

__all__ = ['_anderson_darling', '_chi_stat_test', '_cramer_von_mises', '_energy_dist', '_epps_singleton', '_fisher_exact_stattest', '_g_stat_test', '_hellinger_distance', '_jensenshannon', '_kl_div', '_ks_stat_test', '_mannwhitneyu_rank', '_mmd_stattest', '_psi', '_t_test2samp', '_tvd_stattest', '_wasserstein_distance_norm', '_z_stat_test']
