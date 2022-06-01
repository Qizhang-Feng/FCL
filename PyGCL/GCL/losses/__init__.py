from .jsd import JSD, DebiasedJSD, HardnessJSD
from .vicreg import VICReg
from .infonce import InfoNCE, InfoNCESP, DebiasedInfoNCE, HardnessInfoNCE, FairInfoNCE
from .triplet import TripletMargin, TripletMarginSP
from .bootstrap import BootstrapLatent
from .barlow_twins import BarlowTwins
from .losses import Loss

__all__ = [
    'Loss',
    'InfoNCE',
    'FairInfoNCE',
    'InfoNCESP',
    'DebiasedInfoNCE',
    'HardnessInfoNCE',
    'JSD',
    'DebiasedJSD',
    'HardnessJSD',
    'TripletMargin',
    'TripletMarginSP',
    'VICReg',
    'BarlowTwins'
]

classes = __all__
