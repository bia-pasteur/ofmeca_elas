"""Configuration of the analysis"""
# pylint disable=invalid-name

from dataclasses import dataclass

import numpy as np


@dataclass
class GeneralParams:
    """General parameters of the analysis"""

    results_dir: str


@dataclass
class FistaParams:
    """FISTA parameters"""

    num_iter: int
    num_warp: int
    alpha: float | np.ndarray
    beta: float | np.ndarray
    eps: float
    num_pyramid: int
    pyramid_downscale: float
    pyramid_min_size: int
    mask: np.ndarray | None = None
    outside_multiplier: float = 10.0


@dataclass
class HSParams:
    """Horn Schunck parameters"""

    num_iter: int
    num_warp: int
    alpha: float
    eps: float
    num_pyramid: int
    pyramid_downscale: float
    pyramid_min_size: int
    w: float


@dataclass
class FarnebackParams:
    """Farneback parameters"""

    winSize: int
    pyrScale: float
    numLevels: int
    fastPyramids: bool
    numIters: int
    polyN: int
    polySigma: float
    flags: int


@dataclass
class TVL1Params:
    """TV-L1 parameters"""

    attachment: float
    tightness: float
    num_warp: int
    num_iter: int
    tol: float
    prefilter: bool


@dataclass
class ILKParams:
    """ILK parameters"""

    radius: float
    num_warp: int
    gaussian: bool
    prefilter: bool


@dataclass
class PIVParams:
    """PIV parameters"""

    window_size: int
    overlap: int
    search_area: int
    s2n_thresh: float
    method: str


@dataclass
class OpticalFlowParams:
    """Optical Flow parameters"""

    global_flow: bool
    fista: FistaParams
    hs: HSParams
    farneback: FarnebackParams
    tvl1: TVL1Params
    ilk: ILKParams
    piv: PIVParams


@dataclass
class FistaParamsList:
    """FISTA parameters"""

    num_iter: list[int]
    num_warp: list[int]
    alpha: list[float]
    beta: list[float]
    eps: float
    num_pyramid: int
    pyramid_downscale: float
    pyramid_min_size: int


@dataclass
class HSParamsList:
    """Horn Schunck parameters"""

    num_iter: list[int]
    num_warp: list[int]
    alpha: list[float]
    eps: float
    num_pyramid: int
    pyramid_downscale: float
    pyramid_min_size: int
    w: float


@dataclass
class FarnebackParamsList:
    """Farneback parameters"""

    winSize: list[int]
    pyrScale: list[float]
    numLevels: list[int]
    fastPyramids: bool
    numIters: list[int]
    polyN: list[int]
    polySigma: list[float]
    flags: int


@dataclass
class TVL1ParamsList:
    """TV-L1 parameters"""

    attachment: list[float]
    tightness: list[float]
    num_warp: list[int]
    num_iter: list[int]
    tol: float
    prefilter: bool


@dataclass
class ILKParamsList:
    """ILK parameters"""

    radius: list[float]
    num_warp: list[int]
    gaussian: bool
    prefilter: bool


@dataclass
class PIVParamsList:
    """PIV parameters"""

    window_size: list[int]
    overlap: list[int]
    search_area: list[int]
    s2n_thresh: list[float]
    method: str


@dataclass
class OpticalFlowParamsList:
    """Optical Flow parameters"""

    global_flow: bool
    fista_list: FistaParamsList
    hs_list: HSParamsList
    farneback_list: FarnebackParamsList
    tvl1_list: TVL1ParamsList
    ilk_list: ILKParamsList
    piv_list: PIVParamsList


@dataclass
class ElasticExperiment:
    """Configuration for an experiment on synthetic images of elastic cells"""

    of_funcs: list[str] | str
    vmaxstrain: float
    scale_flow: float
    step_flow: int
    scale_traction: float
    step_traction: int
    T_for_plot: float
    E_for_plot: float
    nu_for_plot: float
    threshold_inf: float
    threshold_sup: float
    scatter_comparison: bool
    T: float | None = None
    E: float | None = None
    nu: float | None = None
    exp_ind: int | None = None
    image_id: str | None = None
    implot: int | None = None


@dataclass
class RegExperiment:
    """Configuration for the regularization testing experiment"""

    of_funcs: list[str] | str
    T: float
    E: float
    nu: float
    factors: list[float]


@dataclass
class NoiseExperiment:
    """Configuration for the noise experiment"""

    of_funcs: list[str] | str


@dataclass
class MicroExperiment:
    """Configuration for an experiment on a microscopy image"""

    im: int
    of_funcs: list[str] | str
    path: str
    active_contour: bool
    E: float
    nu: float
    vmaxstrain: float
    scale_flow: float
    step_flow: int
    scale_traction: float
    step_traction: int
    qt: bool
    vminpositions: float | None = None
    vmaxpositions: float | None = None
    alphapositions: float | None = None
    center_circle_seg: tuple[float, float] | None = None
    radius_circle_seg: float | None = None
    alpha: list[float] | None = None
    beta: list[float] | None = None
    gamma: list[float] | None = None

    def __post_init__(self):
        if isinstance(self.of_funcs, str):
            self.of_funcs = [self.of_funcs]
