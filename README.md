# MD-FDRI

MD-FDRI stands for Map-based, Differential, Fourier Domain Regularized Inversion. MD-FDRI is a framework for
 Single-Pixel Imaging (SPI) applicable at high resolutions and high compression. MDFDRI code accompanies the Opt. Express 30, 22730, 2022 paper
by R.Stojek, A. Pastuszczak, P. Wróbel and R. Kotyński on single-pixel imaging at high resolutions
https://doi.org/10.1364/OE.460025
https://doi.org/10.6084/m9.figshare.19863556

The MDFDRI class definition is included in the mdfdri.py file. 
example_mdfdri_animation.py is an example program which demonstrates the use of the MDFRI class. It runs a simulation of a sequence of SPI measurements with compressive measurements conducted through a varied aperture. As a result it produces an animation and figures with 
examples of the sampling functions and image maps which were used.

MDFDRI needs huge (2x7GB) image sampling and reconstruction matrices, which we will first try to load from the current
 directory, secondly to download from a repository, and third to recalculate. We recommend having at least 128GB memory
 and a reasonable swap file for matrix recalculation. A fast SDD drive and 32GB RAM is needed for executing this 
 example program.
