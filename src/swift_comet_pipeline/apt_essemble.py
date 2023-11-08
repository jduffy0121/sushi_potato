#!/usr/bin/env python3

import random
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from astropy.visualization import (
    ZScaleInterval,
)
from astropy.stats import sigma_clip
from typing import Tuple

from tqdm import tqdm

import os
import sys
import pathlib

from get_mask import build_img_mask

from pipeline_files import PipelineFiles
from configs import read_swift_project_config
from tui import stacked_epoch_menu
from epochs import Epoch
from argparse import ArgumentParser
from stacking import StackingMethod, center_image_on_coords, determine_stacking_image_size

from swift_data import SwiftData
from uvot_image import SwiftUVOTImage
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


def plot_apertures(image: SwiftUVOTImage, aperture_list: list) -> None:
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image)

    im1 = ax1.imshow(image, vmin=vmin, vmax=vmax)
    for aperture in aperture_list:
        x = aperture.positions[0]
        y = aperture.positions[1]
        r = aperture.r
        ax1.add_patch(plt.Circle((x, y), r, fill=False, edgecolor="red"))
    fig.colorbar(im1)
    plt.show()

def plot_image(image: SwiftUVOTImage) -> None:
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image)

    im1 = ax1.imshow(image, vmin=vmin, vmax=vmax)
    fig.colorbar(im1)
    plt.show()

def get_weighted_average_and_std(input_list: list) -> Tuple[float, float]:
    weights = [x.final_apt.r ** 2 for x in input_list]
    avgs = [x.final_stats.mean for x in input_list]
    mu = np.average(avgs, weights = weights)
    std = np.sqrt(np.cov(avgs, aweights = weights))
    return mu, std

def clip_results(list_to_clip: list, sigma: int) -> list[CircularAperture]:
    mu, std = get_weighted_average_and_std(input_list=list_to_clip)
    clipped_results = [x for x in list_to_clip if x.final_stats.mean <= sigma * std + mu and x.final_stats.mean >= sigma * std - mu]
    return clipped_results
    
def get_aperture_stats(image: SwiftUVOTImage, aperture: CircularAperture) -> Tuple[float, float, float, float, float]:
    aperture_stats = ApertureStats(image, aperture)
    min = aperture_stats.min
    max = aperture_stats.max
    median = aperture_stats.median
    average = aperture_stats.mean
    std = aperture_stats.std
    return min, max, median, average, std

def get_min_aperture(image: SwiftUVOTImage, mask: SwiftUVOTImage, current_aperture: CircularAperture, 
                     aperture_list: list[CircularAperture]) -> CircularAperture:
    min_aperture = current_aperture
    init_aperture_stats = ApertureStats(image, current_aperture)
    std = init_aperture_stats.std
    for aperture in aperture_list:
        test_results = aperture_test(image=image, aperture=aperture, mask=mask)
        aperture_stats = ApertureStats(image, aperture)
        if (test_results[0] == True and aperture_stats.std < std):
                min_aperture = aperture
                std = aperture_stats.std
    return min_aperture

def aperture_walk(image: SwiftUVOTImage, mask: SwiftUVOTImage, intial_aperture: CircularAperture,
                step_size: int, max_steps: int) -> Tuple[CircularAperture, list]:
    
    walking_path = []
    current_aperture = intial_aperture
    step_count = 0
    while (step_count < max_steps):
        x = current_aperture.positions[0]
        y = current_aperture.positions[1]
        radius = current_aperture.r

        walking_path.append(current_aperture)

        # Creates 8 new CircularAperture objects by steping in 8 directions 
        up_aperture = CircularAperture((x, y + step_size), r=radius)
        down_aperture = CircularAperture((x, y - step_size), r=radius)
        right_aperture = CircularAperture((x + step_size, y), r=radius)
        left_aperture = CircularAperture((x - step_size, y), r=radius)
        upper_right_aperture = CircularAperture((x + step_size, y + step_size), r=radius)
        upper_left_aperture = CircularAperture((x - step_size, y + step_size), r=radius)
        lower_right_aperture = CircularAperture((x + step_size, y - step_size), r=radius)
        lower_left_aperture = CircularAperture((x - step_size, y - step_size), r=radius)

        next_aperture = get_min_aperture(image=image, mask= mask, current_aperture=current_aperture, aperture_list=[up_aperture, down_aperture, 
                                        right_aperture, left_aperture, upper_right_aperture, upper_left_aperture, 
                                        lower_right_aperture, lower_left_aperture])
        
        if(current_aperture == next_aperture): # This condition means that a local min was found
            break
        else:
            current_aperture = next_aperture
            step_count += 1
    return current_aperture, walking_path, step_count

def aperture_grow(image: SwiftUVOTImage, mask: SwiftUVOTImage, aperture: CircularAperture, grow_size: int, 
                  max_grow_count: int, dmax_limit: float) -> Tuple[CircularAperture, int]:
    current_aperture = aperture
    grow_count = 0
    while (grow_count < max_grow_count):
        x = current_aperture.positions[0]
        y = current_aperture.positions[1]
        radius = current_aperture.r

        new_aperture = CircularAperture((x, y), r=radius + grow_size)
        test = aperture_test(image=image, aperture=new_aperture, mask=mask)
        if (test[0] == False): # This condition means that the larger radius created an aperture that has left the image
            break

        current_aperture_stats = ApertureStats(image, current_aperture)
        new_aperture_stats = ApertureStats(image, new_aperture)
        if (abs(current_aperture_stats.max - new_aperture_stats.max) > dmax_limit): # This condition means that dmax > dmax_limit (most likely indecating a spike in signal)
            break
        else:
            current_aperture = new_aperture
            grow_count += 1
    return current_aperture, grow_count

def validate_mask(mask: SwiftUVOTImage, aperture: CircularAperture) -> bool:
    aperture_stats = ApertureStats(mask, aperture)
    if (aperture_stats.mean >= 0.95):
        return True
    return False


def aperture_test(image: SwiftUVOTImage, aperture: CircularAperture, mask: SwiftUVOTImage) -> Tuple[bool, str]:
    passed_test = False
    message = None
    mask_is_valid = validate_mask(mask=mask, aperture=aperture)
    aperture_stats = ApertureStats(image, aperture)
    if (aperture_stats.min == 0):
        message = "Min is 0"
    elif (aperture_stats.std == 0):
        message = "Std is 0"
    elif (mask_is_valid == False):
        message = "Outside of Mask"
    else:
        passed_test = True
    return passed_test, message

def intialize_aperture(image: SwiftUVOTImage, radius: int, mask: SwiftUVOTImage) -> CircularAperture:
    proper_x_indexes = []
    proper_y_indexes = []
    # Sums up both colums and rows
    column_sum = image.sum(axis=0)
    row_sum = image.sum(axis=1)
    for i, pixel_val in enumerate(column_sum):
        if (pixel_val != 0):
            proper_x_indexes.append(i)
    for j, pixel_val in enumerate(row_sum):
        if (pixel_val != 0):
            proper_y_indexes.append(j)
    while (True): # Keeps testing apertues untill a valid aperture is created
        # Initalizes a random (x, y) from the allowed coordinate list
        x = random.choice(proper_x_indexes)
        y = random.choice(proper_y_indexes)
        aperture = CircularAperture((x, y), r=radius)
        test_results = aperture_test(image=image, aperture=aperture, mask=mask) # Verifies that the random aperture is valid
        if (test_results[0] == True):
            break

    return aperture

@dataclass
class AptPass:
    walking_path: list[CircularAperture]
    final_apt: CircularAperture
    final_stats: ApertureStats
    r_grow: int

@dataclass
class AptWalkConfig:
    filter_type: str
    init_radius: int
    step_size: int
    max_step: int
    grow_size: int
    max_grow: int
    iterations: int
    sigma_used_for_sigma_clip: int
    pixel_threshold: float
    dmax_limit: float

@dataclass
class PrintConfig:
    print_final_rand_place: bool
    print_final_just_walk: bool
    print_each_alg_walk: bool
    print_final_alg_walk: bool
    print_sub_background: bool
    print_stats: bool

def get_background_apt(image: SwiftUVOTImage, mask: SwiftUVOTImage, walk_config: AptWalkConfig, print_config = None) -> BackgroundResult:
    if not print_config:
        print_config = PrintConfig(print_final_rand_place=False, print_final_just_walk=False, print_each_alg_walk=False, print_final_alg_walk=False, print_sub_background=False, print_stats=False)

    max_pixel = np.max(image)
    image = image / max_pixel
    
    first_pass_results = []
    total_walking_path = []
    print(f"\nStarting {walk_config.iterations} iterations of walk and grow algorithm with initial radius of {walk_config.init_radius} pixels")
    for i in tqdm(range(walk_config.iterations)):
        inital_aperture = intialize_aperture(image=image,radius=walk_config.init_radius, mask=mask)
        walk_result, walking_path, steps_taken = aperture_walk(image=image, mask=mask, intial_aperture=inital_aperture, step_size=walk_config.step_size, max_steps=walk_config.max_step)
        grow_result, growth_taken = aperture_grow(image=image, mask=mask, aperture=walk_result, grow_size=walk_config.grow_size, max_grow_count=walk_config.max_grow, dmax_limit=walk_config.dmax_limit)
        min, max, median, average, std = get_aperture_stats(image=image, aperture=grow_result)
        test_result = AptPass(walking_path=walking_path, final_apt=grow_result, final_stats=ApertureStats(image, grow_result), r_grow = grow_result.r - walk_config.init_radius)
        first_pass_results.append(test_result)
        total_walking_path.append(walking_path)
        if (print_config.print_each_alg_walk):
            plot_apertures(image=image*max_pixel, aperture_list=walking_path)
    
    second_pass_results = [x for x in first_pass_results if x.final_stats.mean <=walk_config.pixel_threshold]
    
    if not second_pass_results:
        return None

    third_pass_results = clip_results(list_to_clip=second_pass_results, sigma=walk_config.sigma_used_for_sigma_clip)
    mu, std = get_weighted_average_and_std(input_list=third_pass_results)
    result = ValueAndStandardDev(value = mu*max_pixel, sigma = std)
    if (print_config.print_final_alg_walk):
        final_walk_list = [x.final_apt for x in third_pass_results]
        plot_apertures(image=image * max_pixel, aperture_list=final_walk_list)
    return BackgroundResult(result, params={}), mu*max_pixel, std

def just_walk(image: SwiftUVOTImage, mask: SwiftUVOTImage, iterations: int, step_size: int, max_steps: int, sigma: int, print_config = None) -> Tuple[float, float]:
    if not print_config:
        print_config = PrintConfig(print_final_rand_place=False, print_final_just_walk=False, print_each_alg_walk=False, print_final_alg_walk=False, print_sub_background=False, print_stats=False)
    max_pixel = np.max(image)
    image = image / max_pixel
    total_walking_path = []
    results_list = []
    low_r = 2
    high_r = 50
    print(f'\nStarting {iterations} iterations of walking algorithm with random sized apertures with a random radius of pixel size {low_r} to {high_r}')
    for i in tqdm(range(iterations)):
        r = random.randrange(low_r, high_r, 2)
        inital_aperture = intialize_aperture(image=image, radius=r, mask=mask)
        walk_result,walking_path,steps_taken = aperture_walk(image=image, mask=mask, intial_aperture=inital_aperture, step_size=step_size, max_steps=max_steps)
        test_result = AptPass(walking_path=walking_path, final_apt=walk_result, final_stats=ApertureStats(image, walk_result), r_grow = None)
        results_list.append(test_result)
        total_walking_path.append(walking_path)
    if(print_config.print_final_just_walk):
        final_walk_list = [x.final_apt for x in results_list]
        plot_apertures(image=image*max_pixel, aperture_list=final_walk_list)
    clipped_list = clip_results(list_to_clip=results_list, sigma=sigma)
    mu, std = get_weighted_average_and_std(input_list=clipped_list)
    return mu*max_pixel, std

def rand_place (image: SwiftUVOTImage, mask: SwiftUVOTImage, iterations: int, sigma: int, print_config = None) -> Tuple[float, float]:
    if not print_config:
        print_config = PrintConfig(print_final_rand_place=False, print_final_just_walk=False, print_each_alg_walk=False, print_final_alg_walk=False, print_sub_background=False, print_stats=False)
    max_pixel = np.max(image)
    image = image / max_pixel
    results_list = []
    low_r = 2
    high_r = 50
    print(f'\nStarting {iterations} iterations of random sized apertures placed on image with a random radius of pixel size {low_r} to {high_r}')
    for i in tqdm(range(iterations)):
        r = random.randrange(low_r, high_r, 2)
        inital_aperture = intialize_aperture(image=image, radius=r, mask=mask)
        test_result = AptPass(walking_path=None, final_apt=inital_aperture, final_stats=ApertureStats(image, inital_aperture), r_grow = None)
        results_list.append(test_result)
    if(print_config.print_final_rand_place):
        final_walk_list = [x.final_apt for x in results_list]
        plot_apertures(image=image*max_pixel, aperture_list=final_walk_list)
    clipped_list = clip_results(list_to_clip=results_list, sigma=sigma)
    mu, std = get_weighted_average_and_std(input_list=clipped_list)
    return mu*max_pixel, std

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



def main():

    args = process_args()

    swift_project_config_path = pathlib.Path(args.swift_project_config)
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1

    pipeline_files = PipelineFiles(swift_project_config.product_save_path)
    epoch_path = stacked_epoch_menu(pipeline_files=pipeline_files)

    uvv_img = load_image(pipeline_files, epoch_path)
    swift_data = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    epoch_product = pipeline_files.stacked_epoch_products[epoch_path]
    epoch_product.load_product()
    epoch = epoch_product.data_product
    uvv_mask = epoch.FILTER == SwiftFilter.uvv
    mask = build_img_mask(swift_data=swift_data, epoch=epoch[uvv_mask])

    # (mu, std)
    ground_truths_uvv_sum = {
        "000_2014_14_Aug": (0.0150387, 8.33952e-06), #(x,y,r) = (1394.5261, 808.6096, 73.137788)
        "001_2014_05_Nov": (0.0220941, 1.34707e-05), #(x,y,r) = (335.79648, 1317.9514, 110.02492)
        "002_2014_19_Dec": (0.0391039, 2.33792e-05), #(x,y,r) = (1068.104, 694.848, 85.679456) #problem child
        "003_2015_28_Apr": (0.0337918, 1.74966e-05), #(x,y,r) = (876.83667, 724.57466, 82.961734)
        "004_2015_19_Jun": (0.0138155, 6.832e-06), #(x,y,r) = (994.48627, 704.9159, 89.291774)
        "005_2015_11_Aug": 1, #(x,y,r) = 
        "006_2015_01_Sep": 1, #(x,y,r) = 
        "007_2016_11_Feb": 1, #(x,y,r) = 
        "008_2016_14_Mar": 1, #(x,y,r) = 
        "009_2016_10_Apr": 1, #(x,y,r) = 
        "010_2016_19_Aug": 1, #(x,y,r) = 
        "011_2016_24_Nov": 1, #(x,y,r) = 
    }

    ground_truths_uvw1_sum = {}
    ground_truths_uvv_median = {}
    ground_truths_uvw1_median = {}
    image_list = [uvv_img]

    ##
    ##

    #####################################################################################################
    #                               Params to tune                                                      #
    #####################################################################################################
    init_radius = [2, 5, 10, 15, 20]
    step_size = [1] #Done
    max_step_size = [500, 600, 700, 800, 900, 1000]
    grow_size = [1] #Done
    max_grow_size = [50, 60, 70, 80, 90, 100]
    dmax_limit_V = [1e-1, 1e-2, 1e-3, 1e-4]
    dmax_limit_UVW1 = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    pixel_threshold_V = [1e6, 2e6, 3e6]
    pixel_threshold_UVW1 = [1e6, 2e6, 3e6]
    iterations = [100, 200, 300, 400, 500]
    sigma = [3] #Done
    #####################################################################################################

    result_list = []

    best_params_V = {}
    best_params_UVW1 = {}
    best_accuracy_V = -1
    best_accuracy_UVW1 = -1

    training_iterations_per_image = 1

    for image in image_list:

        ##
        ## TO DO: Get filter_type from a given image ##

        filter_type = SwiftFilter.uvv
        ground_truth = ground_truths_uvv_sum.get("002_2014_19_Dec")
       

        ##
        ##

        for iteration in range(training_iterations_per_image):
            if (filter_type == SwiftFilter.uvv):
                dmax_limit = random.choice(dmax_limit_V)
                pixel_threshold = random.choice(pixel_threshold_V)
                filter_type_str = "V"
            elif (filter_type == SwiftFilter.uw1):
                dmax_limit = random.choice(dmax_limit_UVW1)
                pixel_threshold = random.choice(pixel_threshold_UVW1)
                filter_type_str = "UVW1"
            else:
                print(f"Unable to reconize filter type.")
            
            config_walk = AptWalkConfig(filter_type=filter_type_str, init_radius=random.choice(init_radius), 
                                                   step_size=random.choice(step_size), grow_size=random.choice(grow_size), dmax_limit=dmax_limit, 
                                                   pixel_threshold=pixel_threshold, max_grow=random.choice(max_grow_size), 
                                                   max_step=random.choice(max_step_size), iterations=random.choice(iterations), 
                                                   sigma_used_for_sigma_clip=random.choice(sigma))
            print(f'\nWalking and growing parameters:\n{asdict(config_walk)}')
            print_config =  PrintConfig(print_final_rand_place=True, print_final_just_walk=True, print_each_alg_walk=False, print_final_alg_walk=True, print_sub_background=False, print_stats=True)

            rand_mu, rand_std = rand_place(image=image, mask=mask, iterations= config_walk.iterations, sigma = config_walk.sigma_used_for_sigma_clip, print_config=print_config)
            just_walk_mu, just_walk_std = just_walk(image=image, mask=mask, iterations= config_walk.iterations, max_steps=config_walk.max_step, step_size=config_walk.step_size, sigma = config_walk.sigma_used_for_sigma_clip, print_config=print_config)
            background_result, test_mu, test_std = get_background_apt(image=image, mask=mask, walk_config=config_walk, print_config=print_config)

            if (filter_type_str == "V" and np.abs(1 - test_mu / ground_truth[0]) <= np.abs(1 - best_accuracy_V)):
                best_accuracy_V = test_mu / ground_truth[0]
                best_params_V = asdict(config_walk)
            elif (filter_type_str == "UVW1" and np.abs(1 - test_mu / ground_truth[0]) <= np.abs(1 - best_accuracy_UVW1)):
                best_accuracy_UVW1 = test_mu / ground_truth[0]
                best_params_UVW1 = asdict(config_walk)
            if (print_config.print_stats):
                print(f"\nManual result for this image, aka \"ground truth\" (mu, std): {ground_truth[0]}, {ground_truth[1]}")
                print(f"\nRandom result (mu, std): {rand_mu}, {rand_std}")
                print(f"\tAccuracy (mu, std) (random result/manual results): {rand_mu / ground_truth[0]}, {rand_std / ground_truth[1]}")
                print(f"Just walk result (mu, std): {just_walk_mu}, {just_walk_std}")
                print(f"\tAccuracy (mu, std) (just walk result/manual results): {just_walk_mu / ground_truth[0]}, {just_walk_std / ground_truth[1]}")
                print(f"Alg result (mu, std): {test_mu}, {test_std}")
                print(f"\tAccuracy (mu, std) (alg result/manual results): {test_mu / ground_truth[0]}, {test_std / ground_truth[1]}")         
            if(print_config.print_sub_background):
                rand_img_sub = image - rand_mu
                just_walk_img_sub = image - just_walk_mu
                test_img_sub = image - test_mu
                #plot_image(image=rand_img_sub)
                plot_image(image=image)
                plot_image(image=test_img_sub)

    print(f"\n\nBest V filter accuracy: {best_accuracy_V}")
    print(f"Params that got this accuracy: {best_params_V}")


if __name__ == "__main__":
    main()
