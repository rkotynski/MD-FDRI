#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single pixel imaging at high pixel resolutions.

This code accompanies the Opt. Epxpress 30, 22730, 2022 paper
by R.Stojek, A. PAstuszczak, P. Wróbel and R. Kotyński

 https://doi.org/10.1364/OE.460025
 https://doi.org/10.6084/m9.figshare.19863556)
"""

import numpy as np
import os
from scipy import fft, linalg
import matplotlib.pyplot as plt
import wget
from datetime import datetime

from numba import jit, prange


MATRICES_URL = "https://drive.google.com/uc?export=download&id=1FDLWvgkAB-wnKhTPvZCwffcPZlUTEpnY"

MATRICES_EXTENSION = {'P': '_reconstr.npy', 'M': '_sampling.npy',
                      'Maps': '_maps.npy'}


class MDFDRI:
    """Map-based Differential-Fourier-Domain-Regularized-Inversion class."""

    def __init__(self, matrices_filename=None,
                 lookuptable_filename=None,
                 verbose=False):
        self.verbose = verbose
        self.matrices = dict()
        self.matrices['M'] = None
        self.matrices['P'] = None
        self.matrices['A'] = None
        self.matrices['Maps'] = None
        self.info = 'MDFDRI (DOI:10.1364/OE.460025)\n'
        self.dim = None
        self.lookuptable_filename = "amatrix.npy"\
            if lookuptable_filename is None else lookuptable_filename
        self.matrices_filename = "matrices_768_1024.npz"\
            if matrices_filename is None else matrices_filename

    def create_measurement_matrix(self, dim=(768, 1024), lm=100, m=31):
        """
        Create the binary measurement matrix.

        Create the binary measurement matrix M with rows representing
        sampling patterns.

        Parameters
        ----------
        A : lookup table
        dim : dimension of image maps. The default is (768, 1024).
        lm : number of image maps. The default is 100.

        Returns
        -------
        M : the binary measurement matrix
        image_maps :  an array with image maps
        """
        if self.verbose:
            print("Calculating the measurement matrix")
            print(f"Image Dimensions:{dim}")
            print(f"Number of maps:{lm}")
            print(f"Number of regions per map:{m}")

        if self.matrices["A"] is not None:
            A = self.matrices["A"].copy()
            m = A.shape[1]
        else:
            A = self.lookup_table(m=31)
            self.matrices['A'] = A.copy()
        M = np.zeros((m+1+(lm-1)*(m-1), np.prod(dim)), dtype=np.uint8)
        if self.verbose:
            print(f"Calculating {lm} image maps with {m} regions each.")
        image_maps = np.array([self.create_image_map(m=m, dim=dim)
                              for _ in range(lm)])
        if self.verbose:
            print("Calculating the measurement matrix")
        row = 0
        for i in range(lm):  # loop over image maps
            for j in range(m):  # loop over regions in the map
                M[row:row+A.shape[0], image_maps[i].reshape(-1) == j+1] =\
                    A[:, j].reshape((-1, 1))
            row += A.shape[0]
            if i == 0:
                # first two rows are used with the first image map only
                A = A[2:, :]
        if self.verbose:
            print("OK")
        image_maps -= 1
        return M, image_maps  # return the measurement matrix and image maps

    def lookup_table(self, m=15):
        """Read the lookup table from a file."""
        try:
            print("Trying to read the lookup table from" +
                  f"{self.lookuptable_filename}")
            A = np.load(self.lookuptable_filename,
                        allow_pickle=True)[()][1][m]['A']
            test = self.tst_lookup_table(A)
            if test:
                lookup_table_str = ''.join([(
                    '' if j else '\n')+str(int(A[i, j])) for i in
                    range(A.shape[0])
                    for j in range(A.shape[1])])
                print(f"OK,\nA={lookup_table_str}\n")
                return A
            else:
                print(f"nA={lookup_table_str}\nLookup table test failed\n")
                return False
        except:
            print("Loading lookup table Failed")
        return False

    def tst_lookup_table(self, A=None):
        """Check if the lookup table A fullfills the required conditions.

        Returns: True -ok, False -not ok
        """
        if A is None:
            A = self.matrices['A']
        a = A.astype(float)
        m = a.shape[1]
        if a.shape[0] != m+1:
            print('Wrong shape of the lookup table!')
            return False  # matrix size not ok
        if np.linalg.matrix_rank(np.diff(a, axis=0)) != m:
            print('D(A) is not full rank!')
            return False  # D.A is not full rank
        b = np.vstack((np.diff(np.hstack((np.zeros(
            (m-1, m)), a[2:, :])), axis=0), np.hstack(
                (np.ones((1, m)), -np.ones((1, m))))))
        # is it ok to disregard first two rows of A for subsequent image maps?
        tst = np.linalg.matrix_rank(b) == m-1
        if self.verbose:
            if tst:
                print("Lookup table passed the tests.")
            else:
                print("Lookup table failed to pass the tests!")
        return tst

    def create_image_map(self, dim=(768, 1024), m=31, sgm=None):
        """Create a single image map."""
        """
        Create an image map with integer values in range [1,m]
        based on the uniform quantization of the phase level of spatially
        correlated Gaussian complex-valued noise

        Parameters
        ----------
        dim : dimensions of the image map. The default is (768,1024).
        m : number of pixel regions in the map. The default is 31.
        sgm: correlation width. The default is
        max(np.random.random()*.1, 0.004)
        Returns: an image map with integer values in range [1,m]
        """

        if sgm is None:
            sgm = .01 + .99 * np.random.rand()**2
            # random value in range [.01,.1]
        x, y = np.meshgrid(
            np.linspace(-.5, .5, dim[1]), np.linspace(-.5, .5, dim[0]))
        # Generate complex correlated Gaussian zero-mean noise
        u = np.fft.fft2(np.fft.fftshift(
            np.exp(-x**2/(2*sgm**2)-y**2/(2*sgm**2))))
        u[0, 0] = 0
        u = np.fft.ifft2(np.fft.fft2(np.random.randn(
            *dim) + 1j * np.random.randn(*dim))*u)
        # Find the phase of the Gaussian noise
        new_map = np.angle(u).reshape(-1)
        # Discretize the phase into m levels
        new_map[np.argsort(new_map)] = (m*np.arange(new_map.size
                                                    ))//new_map.size
        new_map = np.array(new_map.reshape(dim), dtype=np.uint8)
        # randomly permute the pixel regions  (for better visualisation only)
        new_map = 1+np.random.permutation(m)[new_map]
        return new_map  # return a map of shape dim with values [1..m]

    def show_image_maps(self, map_no_lst=None, fig_fname=None):
        """Plot a selected image maps.

        map_no_lst is a list of map indices
        (defaults to 5 randomly chosen maps)
        """
        if map_no_lst is None:
            n = 5
            map_lst = [np.random.randint(0, len(self.matrices['Maps'])-1
                                         ) for i in range(n)]
        else:
            n = len(map_no_lst)
            map_lst = map_no_lst
        map_lst = sorted(list(set(map_lst)))

        fig, axs = plt.subplots(1, n, figsize=(5*n, 5))
        for i in range(n):
            map_no = map_lst[i]
            drawn_map = self.matrices['Maps'][map_no]
            m = drawn_map.max()+1
            cmap = plt.cm.get_cmap('RdYlBu', m)
            ax = axs[i]
            pcm = ax.imshow(drawn_map+1, cmap=cmap, vmin=1, vmax=m,
                            origin='lower')
            ax.set_title(f"Image map {map_no}/{len(self.matrices['Maps'])}")
            fig.colorbar(pcm, ax=ax, orientation='horizontal').set_label(
                f'Subsets of pixels ({m} regions)')
        fig.show()
        if fig_fname is not None:
            fig.savefig(fig_fname)

    def show_sampling_patterns(self, patterns_no_lst=None, fig_fname=None):
        """Plot a selected sampling patterns.

        map_no_lst is a list of map indices
        (defaults to 5 randomly chosen maps)
        """
        if patterns_no_lst is None:
            n = 5
            patterns_lst = [np.random.randint(
                0, self.matrices['M'].shape[0]-1) for i in range(n)]
        else:
            n = len(patterns_no_lst)
            patterns_lst = patterns_no_lst
        patterns_lst = sorted(list(set(patterns_lst)))
        fig, axs = plt.subplots(1, n, figsize=(5*n, 5))
        for i in range(n):
            pattern_no = patterns_lst[i]
            drawn_map = self.matrices['M'][pattern_no].reshape(self.dim)
            ax = axs[i]
            ax.imshow(drawn_map,  vmin=0, vmax=1,
                      origin='lower')
            ax.set_title("Binary sampling pattern " +
                         f"{pattern_no}/{self.matrices['M'].shape[0]}")
        fig.show()
        if fig_fname is not None:
            fig.savefig(fig_fname)

    def save_matrices(self):
        """Save calculated matrices to files."""
        if self.verbose:
            print(f"Saving the matrices to {self.matrices_filename}")
        info = f'Timestamp: {datetime.now()}\n'+self.info
        info += "Field description:\nM-measurement matrix" +\
            "\nP-reconstruction matrix\nA-lookup table" +\
                "\nMaps-image maps\ndim-image dimensions"
        np.savez_compressed(self.matrices_filename, M=self.matrices['M'],
                            P=self.matrices['P'], Maps=self.matrices['Maps'],
                            A=self.matrices['A'], dim=self.dim, info=info)

    def download_matrices_from_repository(self):
        """Try to download matrices from repository."""
        print(f"Download precalculated matrices from {MATRICES_URL}")

        answer = input("[y/n]")
        if not(answer in ['y', 'Y']):
            return None
        url = MATRICES_URL
        try:
            print(f"Downloading {url} to {self.matrices_filename}")
            file_name = wget.download(url=url, out=self.matrices_filename)

            self.read_matrices_from_files()
        except:
            print("Downloading failed!!")
            print("Please try to manually download  matrices" +
                  " from {MATRICES_URL}\n\n\n")
        return None

    def precalculate_missing_matrices(self, save=True):
        """Calculate matrices that have not been read from files."""
        newly_calculated = False
        if not('M' in self.matrices.keys()) or\
            not('Maps' in self.matrices.keys()) or\
                self.matrices['Maps'] is None or\
                self.matrices['M'] is None:
            M, Maps = self.create_measurement_matrix()
            self.matrices['M'] = M
            self.matrices['Maps'] = Maps
            self.matrices['P'] = None
            newly_calculated = True
            # if save and newly_calculated:
            #    self.save_matrices()
        if self.verbose:
            print("Fixing matrix representation...")
        self.matrices['Maps'] = list(self.matrices['Maps'])
        self.matrices['Maps'] = [np.array(mp, order='F')
                                 for mp in self.matrices['Maps']]
        self.matrices['maps'] = np.array(
            [Map.ravel() for Map in self.matrices['Maps']], dtype=np.uint8)
        self.matrices['M'] = np.array(self.matrices['M'], dtype=np.float32,
                                      order='F')
        self.dim = self.matrices['Maps'][0].shape
        if not('P' in self.matrices.keys()) or\
                self.matrices['P'] is None:
            self.matrices['P'] = self.reconstruction_matrix()
            newly_calculated = True
        self.tst_lookup_table()
        if save and newly_calculated:
            self.save_matrices()
        return self.matrices

    def reconstruction_matrix(self,   ep=1e-7, mi=0.5, tol=1e-7,
                              use_pinv=False):
        """Calculate the reconstruction matrix.

        pinv - =True to use pseudoinverse or False for FDRI
        ep,mi,tol -parameters passed to fdri
        """
        if self.verbose:
            print('Begining to prepare the reconstruction matrix\n')

        if use_pinv:
            if self.verbose:
                print('Moore-Penrose pseudoinverse\n')
            P = self.diff_op(np.linalg.pinv(self.diff_op(
                self.matrices['M']), rcond=tol), right=True)
        else:
            P = self.dfdri(mi=mi, ep=ep, tol=tol)
        if self.verbose:
            print('Finished calculating the measurement matrix\n')
        return np.array(P, dtype=np.single, order='F')

    def diff_op(self, y, right=False):
        """Discrete gradient operator acting from the left or right side."""
        return np.hstack((np.zeros(
            (y.shape[0], 1)), y))-np.hstack(
            (y, np.zeros((y.shape[0], 1)))) if right else np.diff(
                y, axis=0)

    def dfdri(self, ep=1e-8, mi=0.5, tol=1e-7):
        """Calculate the generalised inverse of the measurement matrix.

        Calculates the Differential Fourier Domain Regularized Inversion
        DFDRI:

          https://10.1364/OE.433199
          Opt. Express 29, 26685-26700 (2021)

          Opt. Express 26, 20009, (2018),
          http://dx.doi.org/10.1364/OE.26.020009

        Input parameters:
         mi - FDRI parameter (defaults to mi=0.5)
         ep - FDRI parameter (defaults to ep=1e-5)

         Output parameters:
         P - the generalized inverse matrix (calculated with Eq. (7) or (8))
        """
        M = self.diff_op(self.matrices['M'])
        if self.verbose:
            print("Calculating the Fourier-domain regularized generalized" +
                  "inversion matrix\n")
        (Ny, Nx) = self.dim

        def w(N): return (2*np.pi/N) * \
            np.hstack((np.arange(N//2), np.arange(-N//2, 0)))
        (wx, wy) = np.meshgrid(w(Nx), w(Ny))
        D = (1/np.sqrt((1-mi)**2 * (np.sin(wx)**2+np.sin(wy)
                                    ** 2) + ep + mi**2*(wx**2+wy**2
                                                        )/(2*np.pi**2)))

        def row_fft2(X): return fft.fftn(
            X.reshape((-1, Ny, Nx)), axes=(-2, -1)).reshape((-1, Ny*Nx))

        def row_ifft2(X): return fft.ifftn(
            X.reshape((-1, Ny, Nx)), axes=(-2, -1)).reshape((-1, Ny*Nx))

        def col_fft2(X): return fft.fftn(X.T.reshape(
            (-1, Ny, Nx)), axes=(-2, -1)).reshape((-1, Ny*Nx)).T
        def col_ifft2(X): return fft.ifftn(X.T.reshape(
            (-1, Ny, Nx)), axes=(-2, -1)).reshape((-1, Ny*Nx)).T

        def FILT_R(X): return row_fft2(row_ifft2(X)*D.reshape(-1))  # F*D*F'*X
        def FILT_L(X): return col_fft2(
            D.reshape(-1, 1)*col_ifft2(X))  # F*D*F'*X
        a = FILT_R(M.reshape((-1, Nx*Ny))).real
        P = FILT_L(linalg.pinv(a, cond=tol)).real
        return self.diff_op(P, right=True)

    def read_matrices_from_files(self):
        """Attampt to read matrices from local files."""
        if not os.path.exists(self.matrices_filename):
            print(f"Matrix file {self.matrices_filename} not found")
            return False
        try:
            print(f"Reading matrices from file: {self.matrices_filename}...")
            mat = np.load(self.matrices_filename)
            print("Reading the measurement matrix")
            self.matrices['M'] = mat['M']
            self.dim = mat['dim']
            print("Reading the reconstruction matrix")
            self.matrices['P'] = mat['P']
            self.matrices['A'] = mat['A']
            self.tst_lookup_table()
            print("Reading image maps")
            self.matrices['Maps'] = mat['Maps']
            self.matrices['maps'] = np.array(
                [Map.ravel() for Map in self.matrices['Maps']], dtype=np.uint8)

            print("OK")
            return True
        except:
            print("Reading failed")
        return False

    def reconstr_algorithm(self, y, dcTol=.00025):
        """Reconstruct the image from the measurment y."""
        x0 = MDFDRI.nmbReconstruct(y, P=self.matrices['P'],
                                   maps=self.matrices['maps'],
                                   dcTol=dcTol,
                                   m=self.matrices['A'].shape[1])
        return x0

    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True)
    def nmbReconstruct(y, P, maps,  dcTol=.00025, m=31):
        """Reconstruct the image from the measurement y (using numba)."""
        x0 = P@y
        mvals, dcTol1 = R_mvals(x0, maps, m=m)
        if dcTol == 0:
            dcTol = dcTol1
        else:
            dcTol *= x0.max()
        for j in prange(x0.shape[0]):
            if x0[j] < dcTol:  # or np.take(mvals,maps[:,j]).min()<dcTol:
                x0[j] = 0
        for k in range(maps.shape[0]):
            fm = maps[k].reshape(1, -1)
            mvals1, qq = R_mvals(x0, fm, m)
            err = np.abs(mvals1-mvals[k]).max()
            if err < dcTol:
                continue
            c = np.ones((mvals1.shape[1],), dtype=mvals1.dtype)
            for lp in range(mvals1.shape[1]):
                if mvals1[0, lp] > 0:  # dcTol:
                    c[lp] = mvals[k, lp]/mvals1[0, lp]
            for j in prange(fm.shape[1]):
                x0[j] *= c[fm[0, j]]
        return x0


@jit(nopython=True, fastmath=True, parallel=True)
def R_mvals(x0, maps, m):
    """Calculate mean values over all regions of all maps."""
    nm, n = maps.shape
    mv = np.zeros((nm, m), dtype=x0.dtype)
    for j in prange(nm):
        mvj = mv[j]
        fmj = maps[j]
        for i in prange(n):
            mvj[fmj[i]] += x0[i]
    if mv.min() > 0:
        return mv, 0
    mn = mv.sum(axis=1).mean()
    nt = 0
    ct = 0

    for j in range(nm):
        c = 0
        for i in range(m):
            if mv[j, i] <= 0:
                c += mv[j, i]
                mv[j, i] = 0
                nt += 1
        if c < 0:
            mv[j, :] *= mn/(mn-c)
            ct -= c
    return mv, 1  # ct/nt


def main():
    """MDFDRI class definition."""
    print("This is the MDFDRI class definition Python 3 file.")
    print("MD-FDRI stands for Map-based, Differential, Fourier Domain")
    print("Regularized Inversion. MD-FDRI is a framework for")
    print(" Single-Pixel Imaging.")
    print("\n")
    print("See the example_mdfdri_animation.py program for MDFDRI usage.")
    print("MDFDRI code accompanies the Opt. Express 30, 22730, 2022 paper")
    print("by R.Stojek, A. Pastuszczak, P. Wróbel and R. Kotyński")
    print("on single-pixel imaging at high resolutions")
    print("https://doi.org/10.1364/OE.460025")
    print("https://doi.org/10.6084/m9.figshare.19863556")


if __name__ == "__main__":
    main()
