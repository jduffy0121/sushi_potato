#!/usr/bin/env python3

import random
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from astropy.visualization import (
    ZScaleInterval,
)
from astropy.stats import sigma_clip
from astropy.io import fits
from typing import Tuple, Optional

from tqdm import tqdm

import os
import sys
import pathlib

from pipeline_files import PipelineFiles
from configs import read_swift_project_config
from tui import stacked_epoch_menu
from swift_data import SwiftData
from epochs import Epoch
from argparse import ArgumentParser
from stacking import StackingMethod, center_image_on_coords, determine_stacking_image_size

from uvot_image import SwiftUVOTImage, SwiftPixelResolution, PixelCoord
from swift_filter import SwiftFilter
from error_propogation import ValueAndStandardDev

from determine_background import BackgroundResult

from dataclasses import dataclass, asdict

from photutils.aperture import (
    CircularAperture,
    ApertureStats,
)

def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    # parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="increase verbosity level"
    )
    parser.add_argument(
        "swift_project_config",
        nargs="?",
        help="Filename of project config",
        default="config.yaml",
    )

    args = parser.parse_args()

    return args


def load_image(pipeline_files: PipelineFiles, epoch_path: Epoch) -> Tuple[SwiftUVOTImage, SwiftUVOTImage]:
    
    if epoch_path is None:
        print("No stacked images found! Exiting.")
        return 1
    if pipeline_files.stacked_epoch_products is None:
        print(
            "Pipeline error! This is a bug with pipeline_files.stacked_epoch_products!"
        )
        return 1

    # pipeline_files.stacked_epoch_products[epoch_path].load_product()
    # epoch = pipeline_files.stacked_epoch_products[epoch_path].data_product

    stacking_method = StackingMethod.summation

    if pipeline_files.stacked_image_products is None:
        print(
            "Pipeline error! This is a bug with pipeline_files.stacked_image_products!"
        )
        return 1
    uw1_sum_prod = pipeline_files.stacked_image_products[
        epoch_path, SwiftFilter.uw1, stacking_method
    ]
    uw1_sum_prod.load_product()
    uw1_sum = uw1_sum_prod.data_product.data

    uvv_sum_prod = pipeline_files.stacked_image_products[
        epoch_path, SwiftFilter.uvv, stacking_method
    ]
    uvv_sum_prod.load_product()
    uvv_sum = uvv_sum_prod.data_product.data

    return uvv_sum

def build_img_mask(
    swift_data: SwiftData,
    epoch: Epoch,
    ) -> SwiftUVOTImage:
    """
    Blindly takes every entry in the given Epoch and attempts to stack it - epoch should be pre-filtered because
    no checks are made here
    """

    # determine how big our stacked image needs to be
    stacking_image_size = determine_stacking_image_size(
        swift_data=swift_data,
        epoch=epoch,
    )

    if stacking_image_size is None:
        print("Could not determine stacking image size!  Not stacking.")
        return None

    mask_list = []

    stacking_progress_bar = tqdm(epoch.iterrows(), total=len(epoch), unit="images")
    for _, row in stacking_progress_bar:
        obsid = row["OBS_ID"]

        image_path = swift_data.get_uvot_image_directory(obsid=obsid) / row["FITS_FILENAME"]  # type: ignore

        exp_time = float(row["EXPOSURE"])

        # read the image
        image_data = fits.getdata(image_path, ext=row["EXTENSION"])

        # center the comet
        image_data = center_image_on_coords(
            source_image=image_data,  # type: ignore
            source_coords_to_center=PixelCoord(x=row["PX"], y=row["PY"]),  # type: ignore
            stacking_image_size=stacking_image_size,  # type: ignore
        )
        mask = image_data != 0
        mask_list.append(mask)       
        stacking_progress_bar.set_description(
            f"{image_path.name} extension {row.EXTENSION}"
        )
    final_mask = np.all(mask_list, axis=0)
    return final_mask

def plot_image(image: SwiftUVOTImage) -> None:
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image)

    im1 = ax1.imshow(image, vmin=vmin, vmax=vmax)
    fig.colorbar(im1)
    plt.show()


def main():

    args = process_args()

    swift_project_config_path = pathlib.Path(args.swift_project_config)
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1

    pipeline_files = PipelineFiles(swift_project_config.product_save_path)
    
    epoch_path = stacked_epoch_menu(pipeline_files=pipeline_files)

    #uvv_img = load_image(pipeline_files, epoch_path)
    swift_data = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    epoch_product = pipeline_files.stacked_epoch_products[epoch_path]
    epoch_product.load_product()
    epoch = epoch_product.data_product
    uvv_mask = epoch.FILTER == SwiftFilter.uvv
    mask = build_img_mask(swift_data=swift_data, epoch=epoch[uvv_mask])
    plot_image(image=mask)


if __name__ == "__main__":
    main()
