from typing import Optional, TypeAlias

import numpy as np
from astropy.time import Time

from configs import read_swift_pipeline_config
from solar_spectrum import solar_count_rate_in_filter
from reddening_correction import reddening_correction, DustReddeningPercent
from error_propogation import ValueAndStandardDev
from count_rate import CountRate


__all__ = [
    "Flux",
    "beta_parameter",
    "OH_flux_from_count_rate",
]

Flux: TypeAlias = ValueAndStandardDev


def beta_parameter(
    dust_redness: DustReddeningPercent, solar_spectrum_time: Optional[Time] = None
) -> float:
    if solar_spectrum_time is None:
        # arbitrary date, happy birthday Mom
        solar_spectrum_time = Time("2016-08-08")

    spc = read_swift_pipeline_config()
    if spc is None:
        print("Could not read pipeline configuration!")
        # TODO: do something better with the error handling
        # TODO: just change this function to return an Optional?
        return 0

    solar_count_rate_in_uw1 = solar_count_rate_in_filter(
        solar_spectrum_path=spc.solar_spectrum_path,
        solar_spectrum_time=solar_spectrum_time,
        effective_area_path=spc.effective_area_uw1_path,
    )
    solar_count_rate_in_uvv = solar_count_rate_in_filter(
        solar_spectrum_path=spc.solar_spectrum_path,
        solar_spectrum_time=solar_spectrum_time,
        effective_area_path=spc.effective_area_uvv_path,
    )

    # print(f"solar count rate in uw1: {solar_count_rate_in_uw1}")
    # print(f"solar count rate in uvv: {solar_count_rate_in_uvv}")
    beta_pre_reddening = solar_count_rate_in_uw1 / solar_count_rate_in_uvv
    beta = (
        reddening_correction(
            effective_area_uw1_path=spc.effective_area_uw1_path,
            effective_area_uvv_path=spc.effective_area_uvv_path,
            dust_redness=dust_redness,
        )
        * beta_pre_reddening
    )

    return beta


def OH_flux_from_count_rate(
    uw1: CountRate,
    uvv: CountRate,
    beta: float,
) -> Flux:
    # TODO: read Lucy's thesis and figure out why the conversion units are 1.275e-12
    # TODO: start tagging with astropy units to avoid conversion units?
    alpha = 1.2750906353215913e-12

    oh_flux = alpha * (uw1.value - beta * uvv.value)

    oh_flux_err = alpha * np.sqrt(uw1.sigma**2 + (uvv.sigma * beta) ** 2)

    return Flux(value=oh_flux, sigma=oh_flux_err)
