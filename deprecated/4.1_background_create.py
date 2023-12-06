#!/usr/bin/env python3

import os
import pathlib
import sys
from astropy.io import fits
import logging as log
import random
import time
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from astropy.visualization import (
    ZScaleInterval,
)
from astropy.stats import sigma_clip
from typing import Tuple

from uvot_image import SwiftUVOTImage
from swift_filter import SwiftFilter

from photutils.aperture import (
    CircularAperture,
    ApertureStats,
)

__version__ = "0.0.1"

# ======================= Algorithm Methods ========================
#
# get_min_aperture(image: SwiftUVOTImage, current_aperture: CircularAperture, aperture_list: list[CircularAperture]) -> CircularAperture:
# aperture_test(image: SwiftUVOTImage, aperture: CircularAperture) -> Tuple[bool, str]:
# intialize_aperture(image: SwiftUVOTImage, radius: int) -> CircularAperture:
# aperture_walk(image: SwiftUVOTImage, intial_aperture: CircularAperture, walking_path: list, step_size: int, max_steps: 
#               int, print_progress: bool) -> Tuple[CircularAperture, int, list]:
# aperture_grow(image: SwiftUVOTImage, aperture: CircularAperture, grow_size: int, max_grow_count: int, 
#               dmax_limit: float, print_progress: bool) -> Tuple[CircularAperture, int]:
# get_aperture_stats(image: SwiftUVOTImage, aperture: CircularAperture) -> Tuple[float, float, float, float, float]:
# get_final_result(clipped_list: list) -> float:
# print_flagged_image_info(flagged_images: list) -> None:
#
# ==================================================================

# Method to test a list of CircularAperture objects to see which has the minimum std.
#
# Params: 
#   image: SwiftUVOTImage.
#   current_aperture: CircularAperture. An aperture that has already been varified that will be intialized.
#   aperture_list: list[CircularAperture].
#
# Returns the CircularAperture object with the minimum std.

def get_min_aperture(image: SwiftUVOTImage, current_aperture: CircularAperture, 
                     aperture_list: list[CircularAperture]) -> CircularAperture:
    min_aperture = current_aperture
    init_aperture_stats = ApertureStats(image, current_aperture)
    std = init_aperture_stats.std
    for aperture in aperture_list:
        test_results = aperture_test(image=image, aperture=aperture)
        aperture_stats = ApertureStats(image, aperture)
        if (test_results[0] == True and aperture_stats.std < std):
                min_aperture = aperture
                std = aperture_stats.std
    return min_aperture

# Method to test and validate a given CircularAperture.
# 
# Params:
#   image: SwiftUVOTImage.
#   aperture: CircularAperture.
#
# Returns a Tuple with a boolean if the object has been validated and str message on the reason if it failed.

def aperture_test(image: SwiftUVOTImage, aperture: CircularAperture) -> Tuple[bool, str]:
    passed_test = False
    message = None
    aperture_stats = ApertureStats(image, aperture)
    if (aperture_stats.min == 0):
        message = "Min is 0"
    elif (aperture_stats.std == 0):
        message = "Std is 0"
    else:
        passed_test = True
    return (passed_test, message)

# Method to intialize a starting aperture.
#
# Params:
#   image: SwiftUVOTImage.
#   radius: int.
#
# Returns a valid CircularAperture object.

def intialize_aperture(image: SwiftUVOTImage, radius: int) -> CircularAperture:
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
        test_results = aperture_test(image=image, aperture=aperture) # Verifies that the random aperture is valid
        if (test_results[0] == True):
            break

    return aperture

# Method to walk from the intial aperture to a local minimum.
# 
# Params:
#   image: SwiftUVOTImage.
#   intial_aperture: CircularAperture
#   walking_path: list[CircularAperture].
#   step_size: int
#   max_steps: int
#
# Returns a tuple with the minimized CircularAperture object, total number of steps the aperture took, and the updated walking path.

def aperture_walk(image: SwiftUVOTImage, intial_aperture: CircularAperture, walking_path: list,
                step_size: int, max_steps: int, print_progress: bool) -> Tuple[CircularAperture, int, list]:
    
    walking_path.append(intial_aperture)
    current_aperture = intial_aperture
    step_count = 0
    while (step_count < max_steps):
        x = current_aperture.positions[0]
        y = current_aperture.positions[1]
        radius = current_aperture.r

        # Creates 8 new CircularAperture objects by steping in 8 directions 
        up_aperture = CircularAperture((x, y + step_size), r=radius)
        down_aperture = CircularAperture((x, y - step_size), r=radius)
        right_aperture = CircularAperture((x + step_size, y), r=radius)
        left_aperture = CircularAperture((x - step_size, y), r=radius)
        upper_right_aperture = CircularAperture((x + step_size, y + step_size), r=radius)
        upper_left_aperture = CircularAperture((x - step_size, y + step_size), r=radius)
        lower_right_aperture = CircularAperture((x + step_size, y - step_size), r=radius)
        lower_left_aperture = CircularAperture((x - step_size, y - step_size), r=radius)

        next_aperture = get_min_aperture(image=image,current_aperture=current_aperture, aperture_list=[up_aperture, down_aperture, 
                                        right_aperture, left_aperture, upper_right_aperture, upper_left_aperture, 
                                        lower_right_aperture, lower_left_aperture])
        
        if(current_aperture == next_aperture): # This condition means that a local min was found
            break
        else:
            current_aperture = next_aperture
            walking_path.append(current_aperture)
            if (print_progress):
                print("Stepping ...")
            step_count += 1
    if (print_progress):
        print(f"Aperture walk concluded in {step_count} step(s).")
    return current_aperture, step_count, walking_path

# Method to grow a aperture until hitting the dmax_limit or going off the image.
#
# Params:
#   image: SwiftUVOTImage.
#   aperture: CircularAperture.
#   max_grow_count: int.
#   dmax_limit: float.
#
# Returns the enlarged aperture and the number of grow iterations as a tuple.

def aperture_grow(image: SwiftUVOTImage, aperture: CircularAperture, grow_size: int, 
                  max_grow_count: int, dmax_limit: float, print_progress: bool) -> Tuple[CircularAperture, int]:
    current_aperture = aperture
    grow_count = 0
    while (grow_count < max_grow_count):
        x = current_aperture.positions[0]
        y = current_aperture.positions[1]
        radius = current_aperture.r

        new_aperture = CircularAperture((x, y), r=radius + grow_size)
        test = aperture_test(image=image, aperture=new_aperture)
        if (test[0] == False): # This condition means that the larger radius means that the aperture has left the image
            break

        current_aperture_stats = ApertureStats(image, current_aperture)
        new_aperture_stats = ApertureStats(image, new_aperture)
        if (abs(current_aperture_stats.max - new_aperture_stats.max) > dmax_limit): # This condition means that dmax > dmax_limit (most likely indecating a star)
            break
        else:
            current_aperture = new_aperture
            if(print_progress):
                print("Growing ...")
            grow_count += 1
    if(print_progress):
        print(f"Aperture grow concluded in {grow_count} step(s).")
    return current_aperture, grow_count

# Method to get the statistics of an aperture.
#
# Params:
#   image: SwiftUVOTImage.
#   aperture: CircularAperture.
#
# Returns a tuple of the min, max, median, average, and std pixel values of the aperture.

def get_aperture_stats(image: SwiftUVOTImage, aperture: CircularAperture) -> Tuple[float, float, float, float, float]:
    aperture_stats = ApertureStats(image, aperture)
    min = aperture_stats.min
    max = aperture_stats.max
    median = aperture_stats.median
    average = aperture_stats.mean
    std = aperture_stats.std
    return min, max, median, average, std

# Method to take the list of sorted param and sigma clips to get the average.
#
# Params:
#   sorted_list: list.
#   sigma: int.
#
# Returns the average pixel value of the param found during all iterations.

def get_final_result(sorted_list: list, sigma: int) -> float:
    condensed_list = []
    clipped_results = sigma_clip(sorted_list,sigma=sigma,maxiters=None,cenfunc=np.mean)
    for result in clipped_results:
        if (result != '--'):
            condensed_list.append(result)
    return np.mean(condensed_list)

# Method to print out the information about every flagged imaged.
#
# Params:
#   flagged_images: list[CircularAperture]
#
# Returns nothing.

def print_flagged_image_info(flagged_images: list[CircularAperture]) -> None:
    for image in flagged_images:
        if (image == None): # Breaks out of loop if the current image is null
            break
        if (image.filter_type == SwiftFilter.uvv):
            filter_type = "V"
        elif (image.filter_type == SwiftFilter.uw1):
            filter_type = "UVW1"
        if (image.stacking_method == StackingMethod.summation):
            image_type = "Summed"
        elif (image.stacking_method == StackingMethod.median):
            image_type = "Median"
        
        print(f"{image_type} {filter_type} filter for tids {image.sources[0][0]} to {image.sources[-1][0]}")

# ======================== Plotting Methods ========================
#
# plot_results(image: SwiftUVOTImage, test_results: list) -> None:
# plot_total_walking_path(image: SwiftUVOTImage, walking_path: list, diff_color: bool) -> None:
# plot_iteration_walking_path(image: SwiftUVOTImage, walking_path: list, iteration: int, diff_color: bool) -> None:
# plot_step_info(test_results: list, max_steps: int, iterations: int) -> None:
# plot_growth_info(test_results: list, max_growth: int, iterations: int) -> None:
#
# ==================================================================

# Method to plot all the results from all iterations.
#
# Params:
#   image: SwiftUVOTImage.
#   test_results: list.
#
# Returns nothing.

def plot_results(image: SwiftUVOTImage, test_results: list) -> None:
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image)

    im1 = ax1.imshow(image, vmin=vmin, vmax=vmax)
    for results in test_results:
        aperture = results[0]
        x = aperture.positions[0]
        y = aperture.positions[1]
        r = aperture.r
        ax1.add_patch(plt.Circle((x, y), r, fill=False, edgecolor="black"))
    fig.colorbar(im1)
    plt.show()

# Method to plot the entire walk path that all iterations take.
#
# Params:
#   image: SwiftUVOTImage.
#   walking_path: list.
#   diff_color: bool.
#
# Returns nothing.

def plot_total_walking_path(image: SwiftUVOTImage, walking_path: list, diff_color: bool) -> None:
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image)

    im1 = ax1.imshow(image, vmin=vmin, vmax=vmax)
    i = 0
    next_is_start = False
    current_end = False
    while i < len(walking_path):
        current_start = next_is_start
        # Sees if the next iteration is either the end or a string placement holder, if so, that means the current pass is
        # either the final aperture in the list or the final aperture in the interation's walk
        try: 
            aperture = walking_path[i + 1]
            x = aperture.positions[0]
        except (AttributeError, IndexError):
            current_end = True
        # Sees if the current iteration is a string placement holder, if so, that means that the next pass is the start
        # of an iteration's walk
        try:
            aperture = walking_path[i]
            x = aperture.positions[0]
            y = aperture.positions[1]
            r = aperture.r
            if (current_start and diff_color): # assigns white circle to start of an iteration's walk
                ax1.add_patch(plt.Circle((x, y), r, fill=False, edgecolor="white"))
                next_is_start = False
            elif (current_end and diff_color): # assigns black circle to end of an iteration's walk
                ax1.add_patch(plt.Circle((x, y), r, fill=False, edgecolor="black"))
                current_end = False
            elif (diff_color): # assigns red circles to all steps of an iteration's walk
                ax1.add_patch(plt.Circle((x, y), r, fill=False, edgecolor="red"))
            else: # assigns all black circles if diff_color = False
                ax1.add_patch(plt.Circle((x, y), r, fill=False, edgecolor="black"))
            i += 1
        except AttributeError:
            next_is_start = True
            i += 1
    fig.colorbar(im1)
    plt.show()

# Method to plot a single iteration's walk path.
#
# Params:
#   image: SwiftUVOTImage.
#   walking_path: list.
#   iteration: int. Current iteration, not total iterations.
#   diff_color: bool.
#
# Returns nothing.

def plot_iteration_walking_path(image: SwiftUVOTImage, walking_path: list, iteration: int, diff_color: bool) -> None:
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image)

    im1 = ax1.imshow(image, vmin=vmin, vmax=vmax)

    start_index = walking_path.index(f"{iteration} walk") + 1 # get starting index
    try:
        end_index = walking_path.index(f"{iteration + 1} walk") - 1 # get ending index if iteration is not the last one
    except ValueError:
        end_index = len(walking_path) - 1 # get ending index if iteration is the last one
    
    # plot inital aperture
    inital_aperture = walking_path[start_index]
    x_init = inital_aperture.positions[0]
    y_init = inital_aperture.positions[1]
    r_init = inital_aperture.r
    if (diff_color):
        ax1.add_patch(plt.Circle((x_init, y_init), r_init, fill=False, edgecolor="white"))
    else:
        ax1.add_patch(plt.Circle((x_init, y_init), r_init, fill=False, edgecolor="black"))

    # plot all steps
    i = start_index + 1
    while i < end_index:
        aperture = walking_path[i]
        x = aperture.positions[0]
        y = aperture.positions[1]
        r = aperture.r
        if (diff_color):
            ax1.add_patch(plt.Circle((x, y), r, fill=False, edgecolor="red"))
        else:
            ax1.add_patch(plt.Circle((x, y), r, fill=False, edgecolor="black"))
        i += 1
    
    # plot final aperture
    final_aperture = walking_path[end_index]
    x_final = final_aperture.positions[0]
    y_final = final_aperture.positions[1]
    r_final = final_aperture.r
    ax1.add_patch(plt.Circle((x_final, y_final), r_final, fill=False, edgecolor="black"))
    fig.colorbar(im1)

    plt.show()

# Method to plot the stats about number of steps that the algorithm ran as a bar graph (min, average, max).
#
# Params:
#   test_results: list.
#   max_steps: int.
#   iterations: int.
#
# Returns nothing.

def plot_step_info(test_results: list, max_steps: int, iterations: int) -> None:
    min_steps_taken = test_results[0][6]
    max_steps_taken = test_results[0][6]
    sum_of_steps = 0
    for result in test_results:
        if (result[6] > max_steps_taken):
            max_steps_taken = result[6]
        if (result[6] < min_steps_taken):
            min_steps_taken = result[6]
        sum_of_steps += result[6]
    average_steps = sum_of_steps / (iterations + 1)
    formatted_average = format(average_steps, '.2f')
    x_values = ['Min Steps Taken', 'Average Steps Taken', 'Max Steps Taken', 'MAX_STEPS']
    y_values = [min_steps_taken, float(formatted_average), max_steps_taken, max_steps]
    fig = plt.figure(figsize = (10, 5))
    plt.bar(x_values, y_values, color = 'blue', width=0.4)
    for i, v in enumerate(y_values):
        plt.text(i - 0.05, v + 0.15, str(v))

    plt.show()

# Method to plot the stats about number of growth steps that the algorithm ran as a bar graph (min, average, max).
#
# Params:
#   test_results: list.
#   max_growth: int.
#   iterations: int.
#
# Returns nothing.

def plot_growth_info(test_results: list, max_growth: int, iterations: int) -> None:
    min_grow_taken = test_results[0][7]
    max_grow_taken = test_results[0][7]
    sum_of_steps = 0
    for result in test_results:
        if (result[7] > max_grow_taken):
            max_grow_taken = result[7]
        if (result[7] < min_grow_taken):
            min_grow_taken = result[7]
        sum_of_steps += result[7]
    average_steps = sum_of_steps / (iterations + 1)
    formatted_average = format(average_steps, '.2f')
    x_values = ['Min Grow Steps Taken', 'Average Grow Steps Taken', 'Max Grow Steps Taken', 'MAX_GROWTH']
    y_values = [min_grow_taken, float(formatted_average), max_grow_taken, max_growth]
    fig = plt.figure(figsize = (10, 5))
    plt.bar(x_values, y_values, color = 'blue', width=0.4)
    for i, v in enumerate(y_values):
        plt.text(i - 0.05, v + 0.15, str(v))

    plt.show()

def main():

    # ======================== Background Selection Parameters ========================
    #
    # init_radius: int (in units of pixels). How big the inital radius will be for each iteration.
    # step_size: int (in units of pixels). How big each step the aperture will take during the walk stage.
    # grow_size: int (in units of pixels). How big the aperture's radius will grow for each iteration during the growing stage.
    # dmax_limit: float (in unit of pixels). The upper bounded limit for the change of the maximum pixel value in the aperture 
    #                                        from the current_aperture_radius to the current_aperture_radius + grow_size.
    # max_grow_count: int (no units). The upper bounded limit for the total number of iterations that an aperture can grow during the growing stage.
    # max_steps: int (no units). The upper bounded limit for the total number of steps that an aperture can take during the walking stage.
    # pixel_threshold: float (in units of pixels). The upper bounded limit for the maximum value the average pixel value can be for a valid aperture,
    #                                              will flag any aperture result that is does not obey this threshold.
    # iterations: int (no units). The total number of iterations that the algorithm will run on each image in image_list.
    # sigma: int (no units). The sigma value used for sigma clipping the results.
    #
    # sorting type: ['average', 'median', 'std', None]. String representing the param that the algoithm will use to determine the background, 
    #                                                      it will take the average param selected.
    #
    # plot_every_iteration: bool. Plots the walking path for every iteration.
    # plot_all_walks: bool. Plots every walking path from every iteration on a single image.
    # plot_test_results: bool. Plots all the final apertures for every iteration on a single image.
    # plot_steps: bool. Plots stats for the number of steps used during the walking stage.
    # plot_growth: bool. Plots stats for the number of growth iterations used during the growing stage.
    #
    # diff_colors: bool. Plots different color apertures (white for init, red for step(s), black for final) in plot_every_iteration and plot_all_walks.
    #
    # print_each_iteration_progess: bool. Allows toggle of printing algorithm progress during each iteration of the walk and grow.
    #
    # ===================================================================================
    
    #fits_path = stacked_fits_path(
    #    stack_dir_path="/Users/jduffy0121/Desktop/AMO/Image_Analysis/Stacking_images/C_2013US10/stacked/000_2014_14_Aug_uvv_sum.fits",
    #    epoch_path="/Users/jduffy0121/Desktop/AMO/Image_Analysis/Stacking_images/C_2013US10/epochs",
    #    filter_type=filter_type,
    #    stacking_method="summed",
    #)
    image = fits.getdata('/Users/jduffy0121/Desktop/AMO/Image_Analysis/Stacking_images/C_2013US10/stacked/000_2014_14_Aug_uvv_sum.fits')

    init_radius = 10
    step_size = 1
    grow_size = 1
    dmax_limit_UVW1 = 0.0006
    pixel_threshold_UVW1 = 1e8
    dmax_limit_V = 0.0028
    pixel_threshold_V = 1e8
    max_grow_count = 250
    max_steps = 100
    iterations = 100
    sigma = 3

    determine_background(image=image, init_radius=init_radius, step_size=step_size, grow_size=grow_size, 
                         dmax_limit=dmax_limit_V, pixel_threshold=pixel_threshold_V, max_grow_count=max_grow_count, max_steps=max_steps,
                         iterations=iterations, sigma=3)

    sorting = ['average','median', 'std', None]
    sorting_type = sorting[0]

    manuel_pixel_values = {"UVW1_sum": 1, "UVW1_median": 1, "V_sum": 1, "V_median": 1}

    plot_every_iteration = False
    plot_all_walks = False
    plot_test_results = True
    plot_steps = False
    plot_growth = True
    
    diff_colors = False

    print_each_iteration_progess = False

    image_list = []
    flagged_images = []
    results_list = []

    start = time.process_time()

    for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
        summed_image = stacked_images[(filter_type, StackingMethod.summation)]
        median_image = stacked_images[(filter_type, StackingMethod.median)]
        image_list.append(summed_image)
        image_list.append(median_image)

    for image_object in image_list:

        test_results = []
        valid_results = []
        walking_path = []
        max_pixel = np.max(image_object.stacked_image)
        median_pixel = np.median(image_object.stacked_image)
        if (image_object.filter_type == SwiftFilter.uvv):
            filter_type = "V"
            image_type = "Summed"
            image = image_object.stacked_image / max_pixel
            dmax_limit = dmax_limit_V
            pixel_threshold = pixel_threshold_V
            if (image_object.stacking_method == StackingMethod.median):
                image_type = "Median"
                manuel_pixel = manuel_pixel_values["V_median"]
            elif (image_object.stacking_method == StackingMethod.summation):
                image_type = "Summed"
                manuel_pixel = manuel_pixel_values["V_sum"]
            else:
                print(f"Unable to reconize filter type.\nPlease try again.")
                continue
        elif (image_object.filter_type == SwiftFilter.uw1):
            filter_type = "UVW1"
            image = image_object.stacked_image / max_pixel
            dmax_limit = dmax_limit_UVW1
            pixel_threshold = pixel_threshold_UVW1
            if (image_object.stacking_method == StackingMethod.median):
                image_type = "Median"
                manuel_pixel = manuel_pixel_values["UVW1_median"]
            elif (image_object.stacking_method == StackingMethod.summation):
                image_type = "Summed"
                manuel_pixel = manuel_pixel_values["UVW1_sum"]
            else:
                print(f"Unable to reconize filter type.\nPlease try again.")
                continue
        else:
            print(f"Unable to reconize filter type.\nPlease try again.")
            continue

        print('----------------------------------------')
        print(f"Image: \'{image_type}\' with \'{filter_type}\' filter for tids \'{image_object.sources[0][0]}\'" + 
              f" to \'{image_object.sources[-1][0]}\'.")

        for iteration in range(iterations):
            if (print_each_iteration_progess):
                print(f"Iteration {iteration + 1} starting ...")
            walking_path.append(f"{iteration + 1} walk") # assigns a string placement holder in the list to differentiate between iteration walks
            if (print_each_iteration_progess):
                print(f"Initializing aperture ...")
            inital_aperture = intialize_aperture(image=image,radius=init_radius)
            if (print_each_iteration_progess):
                print(f"Aperture initalized at ({inital_aperture.positions[0]},{inital_aperture.positions[1]})" +
                    f" with radius {inital_aperture.r}.")
                print("Beginning walk ...")
            walk_result, step_count, walking_path = aperture_walk(image=image,intial_aperture=inital_aperture, 
                                                                                        walking_path=walking_path,step_size=step_size, 
                                                                                        max_steps=max_steps, print_progress=print_each_iteration_progess)
            if (plot_every_iteration):
                plot_iteration_walking_path(image=image,walking_path=walking_path,iteration=iteration + 1, diff_color=diff_colors)
            if (print_each_iteration_progess):
                print(f"Beginning grow ...")
            grow_result, growth_count = aperture_grow(image=image,aperture=walk_result,grow_size=grow_size, 
                                                        max_grow_count=max_grow_count,dmax_limit=dmax_limit, print_progress=print_each_iteration_progess)
            min, max, median, average, std = get_aperture_stats(image=image, aperture=grow_result)
            test_results.append((grow_result, min, max, median, average, std, step_count, growth_count))
            if (print_each_iteration_progess):
                print(f"Iteration {iteration + 1} concluded.")
                print(f"Minimum aperture at ({grow_result.positions[0]},{grow_result.positions[1]})" +
                    f" with radius {grow_result.r}.\n")
            else: # Will print progress for every 10 % of iterations (will delete these print messages afterwards)
                percent_completed = iteration / iterations
                LINE_UP = '\033[1A'
                LINE_CLEAR = '\x1b[2K'
                for value in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    if (percent_completed == value):
                        print(f'{value * 100} % of iteration(s) completed for this image ...')
                        print(LINE_UP, end=LINE_CLEAR)
                if (percent_completed == 1):
                    print(LINE_UP, end=LINE_CLEAR)
        
        for result in test_results:
            if (result[4] <= pixel_threshold): # Test to see if all iterations average pixel value is equal to or less than the pixel_threshold
                valid_results.append(result)

        if not valid_results: # If there are no valid results, flag for manual review and continue to the next image
            flagged_images.append(image_object)
            print(f"Image flagged!\nAll results for this image are above the average pixel threshold of \'{pixel_threshold}\'.\nManually check this image.")
            print('----------------------------------------\n')
            continue
        
        numpy_array = np.array(valid_results)
        if (sorting_type == 'average'):
            values_to_be_sorted = [x[4] for x in numpy_array]
            print('Results clipped by the \'average\' pixel value.')
        elif (sorting_type == 'median'):
            values_to_be_sorted = [x[3] for x in numpy_array]
            print('Results clipped by the \'median\' pixel value.')
        elif (sorting_type == 'std'):
            values_to_be_sorted = [x[5] for x in numpy_array]
            print('Results clipped by the \'standard deviation\' pixel value.')
        else:
            values_to_be_sorted = [x[4] for x in numpy_array]
            print('Results clipping not specified.\nDefault to being clipped by the \'average\' pixel value.')

        result = get_final_result(sorted_list=values_to_be_sorted, sigma=sigma)
        result *= max_pixel
        results_list.append((image_object, result))

        print(f"\nThe algorithm has found the background to have a pixel value of {format(result, '.4f')}.\nThe median pixel value for this image is {format(median_pixel, '.4f')}" 
              + f" (the algortithm result is {format(result/median_pixel * 100, '.4f')}% of this median pixel value).\nThe manuel pixel value for this image is {format(manuel_pixel, '.4f')}" 
              + f" (the algortithm result is {format(result/manuel_pixel * 100, '.4f')}% of this manuel pixel value)")
        print('----------------------------------------\n')

        if (plot_all_walks):
            plot_total_walking_path(image=image,walking_path=walking_path,diff_color=diff_colors)
        if (plot_test_results):
            plot_results(image=image,test_results=test_results)
        if (plot_steps):
            plot_step_info(test_results=test_results,max_steps=max_steps,iterations=iterations)
        if (plot_growth):
            plot_growth_info(test_results=test_results,max_growth=max_grow_count,iterations=iterations)

    end = time.process_time()
    print(f"Algorithm took {format(end - start, '.4f')} seconds to complete {iterations} iteration(s) for {len(image_list)} image(s).")
    print(f"\n{len(flagged_images)} image(s) flagged for manual review.")
    print_flagged_image_info(flagged_images=flagged_images)
    print()

if __name__ == "__main__":
    sys.exit(main())