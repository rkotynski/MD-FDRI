#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make an animation from a SPI simulation with MD-FDRI."""


from mdfdri import MDFDRI
import timeit
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib



def psnr(orig, tstimg, decimal_points=2):
    """Calculate Peak-Signal-to-Noise-Ratio."""
    MSE = np.mean((orig.ravel() - tstimg.ravel()) ** 2)
    vmax = orig.max()
    if MSE > 0 and vmax > 0:
        PSNR = np.round(10 * np.log10(vmax**2 / MSE), decimal_points)
    else:
        PSNR = np.nan
    return PSNR


def masked_image_iter(filename, xyr=None, time_s=15, fps=15, dim=(768, 1024)):
    """Prepare a sequence of images with a moving mask."""
    # xyr - an array defining the locatons and radii of a moving mask(x,y,r)
    # within the image, x is in the range [-0.5,0.5]
    if xyr is None:
        xyr = np.array([(.5, 0, 1.18), (.3, 0, .1), (.3, .3, .1),
                        (.4, .3, .1), (-.4, .3, .1), (-.4, -.3, .1),
                        (.4, -.3, .1), (.5, .4, .1), (0, 0, .5),
                        (0, 0, .1), (-.5, 0, .2), (-.5, 0, .5)])
    # find "distances" between (x,y,r) points
    d = np.hstack((0, np.cumsum(np.linalg.norm(np.diff(xyr, axis=0), axis=1))))

    d_interp = np.sort(
        np.array(list(set(np.linspace(0, d[-1],
                                      max(int(fps*time_s), 1))) | set(d))))
    # interpolate mask locations
    xyr_interp = np.array([np.interp(d_interp, d, xyr[:, i])
                          for i in range(xyr.shape[1])]).T

    whole_imag = np.array(Image.open(filename).resize(
        size=(dim[1], dim[0])), dtype=np.single).reshape(-1)/255
    xm, ym = np.meshgrid(
        np.linspace(-.5, .5, dim[1])*dim[1]/max(dim),
        np.linspace(-.5, .5, dim[0])*dim[0]/max(dim))
    n = xyr_interp.shape[0]

    for i in range(n):
        masked_imag = whole_imag.reshape(-1).copy()
        xc, yc, r = xyr_interp[i]
        mask = ((xm-xc)**2+(ym-yc)**2 <= r**2).reshape(-1)
        masked_imag[~mask] = 0
        yield masked_imag, mask, whole_imag, i, n


def mdfdri_animation_example(mdfdri, fname, animation_fname,
                             fps=15, time_s=15):
    """Make a sequence of MDFDRI measurements and create an animation."""
    print(f"Prepare an animation {animation_fname} by masking parts of")
    print(f"image {fname} and conducting an SPI measurement using MD-FDRI.")
    print(f"FPS={fps}, animation time={time_s}s")
    M = mdfdri.matrices['M']  # Measurement matrix
    k, n = M.shape
    dim = tuple(mdfdri.dim)
    print(f"\nMeasurement matrix size:{k,n}\n" +
          f"Image size:{dim}\nCompression ratio:{round(100*k/n,2)}%\n")
    # Compile numba function
    _ = mdfdri.reconstr_algorithm(np.zeros(k, dtype=np.float32))
    matplotlib.use("Agg")
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax0, ax1, ax2 = ax
    palette = plt.cm.bone.with_extremes(bad='tab:brown', over='y')
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='SPI at high pixel resolutions',
                    artist='University of Warsaw, Faculty of Physics',
                    comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    with writer.saving(fig, animation_fname, 100):
        for masked_imag, mask, whole_imag, i, n in masked_image_iter(
                fname, fps=fps, time_s=time_s):
            print(f'SPI frame {i+1} out of {n}')
            if i == 0:  # Initial frame
                vm = whole_imag.max()*1.3
                # Measure compressively the image x and store the result in y
                y = M@whole_imag

                dt = timeit.default_timer()
                reconstructed_imag = mdfdri.reconstr_algorithm(
                    y)  # Reconstruct the measured image from y
                dt = (timeit.default_timer()-dt)*1e3

                img0 = ax0.imshow(whole_imag.reshape(dim), origin='upper',
                                  vmax=vm, vmin=0, cmap=palette)
                ax0.set_title('Measured image\n(ground truth)')
                img1, = ax1.plot(y, '.')
                ax1.axes.get_yaxis().set_visible(False)
                ax1.set_title(
                    "Compressive measurement\nCompression ratio=" +
                    f"{round(100*M.shape[0]/M.shape[1],2)}%")
                img2 = ax2.imshow(reconstructed_imag.reshape(dim),
                                  origin='upper',
                                  vmax=vm, vmin=0, cmap=palette)
                psnr_criterium = psnr(whole_imag, reconstructed_imag)

                ax2.set_title(
                    "Reconstructed image\n" +
                    f"PSNR={round(psnr_criterium, 1)}dB,\n" +
                    f"reconst. time={round(dt, 2)}ms")
                writer.grab_frame()
            else:  # subsequent frames
                # IMAGE MEASUREMENT AND RECONSTRUCTION
                # Measure compressively the image x and store the result in y
                y = M@masked_imag
                img0.set_data(masked_imag.reshape(dim))
                img1.set_data(np.linspace(1, k, k), y)
                if y.max() > y.min():
                    ax1.axes.set_ylim(y.min(), y.max())
                dt = timeit.default_timer()
                # Reconstruct the measured image from y
                reconstructed_imag = mdfdri.reconstr_algorithm(y)
                dt = (timeit.default_timer()-dt)*1e3
                img2.set_data(reconstructed_imag.reshape(dim))
                psnr_criterium = psnr(
                    whole_imag[mask], reconstructed_imag[mask])

                ax2.set_title(
                    "Reconstructed image\n" +
                    f"PSNR={round(psnr_criterium, 1)}dB,\n" +
                    f"reconst. time={round(dt, 2)}ms")
                writer.grab_frame()


def mdfdri_example(matrices_filename="matrices_768_1024.npz", verbose=True):
    """Show how to use MDFDRI."""
    print("This is an example program showing how to use the MDFDRI class.")
    print("MD-FDRI stands for Map-based, Differential, Fourier Domain")
    print("Regularized Inversion. MD-FDRI is a framework for")
    print(" Single-Pixel Imaging applicable at high resolutions")
    print(" and high compression.")
    print("MDFDRI code accompanies the Opt. Express 30, 22730, 2022 paper")
    print("by R.Stojek, A. Pastuszczak, P. Wróbel and R. Kotyński")
    print("on single-pixel imaging at high resolutions")
    print("https://doi.org/10.1364/OE.460025")
    print("https://doi.org/10.6084/m9.figshare.19863556")
    print("\n")

    print("MDFDRI needs huge (2x9.4GB) image sampling and reconstruction")
    print(" matrices, which we will first try to load from the current")
    print(" directory, secondly to download from a repository, and third to")
    print(" recalculate. We recommend having at least 128GB memory")
    print(" and a reasonable swap file for matrix recalculation. ")
    print(" A fast SDD drive and 32GB RAM is needed for executing this ")
    print(" example program.\n")

    mdfdri = MDFDRI(verbose=verbose, matrices_filename=matrices_filename)
    mdfdri.read_matrices_from_files()
    if mdfdri.matrices['P'] is None:
        print("Matrix reading unsaccesfull,")
        print("trying to download from repository...")
        mdfdri.download_matrices_from_repository()
    if mdfdri.matrices['P'] is None:
        print("Matrix reading unsaccesfull,")
        print("Trying to recalculate...")
        mdfdri.precalculate_missing_matrices()
    if mdfdri.matrices['P'] is None:
        print("Matrix recalculation unsaccesfull,")
        print("Aborting...")
        return

    fig1_fname = "fig_image_maps.jpg"
    print(f"\nPlot sample image maps and save the figure to {fig1_fname}\n")
    mdfdri.show_image_maps(fig_fname=fig1_fname)

    fig2_fname = "fig_sampling_patterns.jpg"
    print(f"Plot some sampling patterns and save the figure to {fig2_fname}\n")
    mdfdri.show_sampling_patterns(fig_fname=fig2_fname)

    return mdfdri


print("Executing mdfdri_animation_example() function.\n")

if ('mdfdri' not in locals()) and (  # Don't read the matrices again if they
        'mdfdri' not in globals()):  # are still in memory
    mdfdri = mdfdri_example()
else:
    print("Using previously read matrices.")


mdfdri_animation_example(mdfdri,
                         fname="FUWchart_768_1024.jpg",
                         animation_fname="mdfdri_animation_768_1024.mp4")
