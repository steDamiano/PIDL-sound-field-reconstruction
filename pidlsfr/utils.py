# This script includes portions of code from the repository "local_soundfield_reconstruction"
# (https://github.com/manvhah/local_soundfield_reconstruction), which is licensed under the MIT License.
# The original license text is included below.

# Copyright (c) 2021 manvhah

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import h5py as hdf
from scipy.spatial import distance
from scipy.special import spherical_jn

c0 = 343
def sinc_kernel_multifreq(
        psize: np.ndarray, 
        K: int, 
        frequencies: np.ndarray, 
        h_dist: float
    ):
    '''
    Generate BL dictionary, whose atoms are drawn from multivariate normal distributions with covariance matrix defined using a Bessel function.

    Parameters
    ----------
    psize: np.ndarray
        2D array containing the number of grid points along the x and y dimension.
    K: int
        Number of dictionary atoms.
    frequencies: np.ndarray
        Array containing the frequency modeled by the dictionary. Each atom represents one of these frequencies.
    h_dist: float
        Horizontal and vertical distance between adjacent pair of points.

    Returns
    -------
    H: np.ndarray
        Generated dictionary
    '''

    x = np.arange(psize[0])
    y = np.arange(psize[1])
    N = psize[0] * psize[1]     # Total number of grid points
    
    xx, yy = np.meshgrid(x,y)
    x_grid = np.array([z for z in zip(xx.flatten(),yy.flatten())])

    mu = np.zeros(N)            # Mean initialization
    H = np.zeros((N,K))         # Dictionary initialization
    
    for k in range(K):
        print(f'Generating sinc dictionary, atom: {k}/{K}')
        
        
        phasedist = 2 * np.pi*frequencies[k] / 343 * h_dist * distance.cdist(x_grid,x_grid)
        sigma  = spherical_jn(0, phasedist)

        H[:,k] = np.random.multivariate_normal(mu, sigma)
        H[:, k] *= 1 / np.linalg.norm(H[:, k])

    np.save(f'sinc_dictionary_{K}_atoms_{frequencies[0]}-{frequencies[-1]}.npy', H)

    return H

def read_hdf(filename, groups = None):
    """
    read data from hdf file
   
    params:
        filename
        groups (default: None) select a specific group

    returns:
        dictionary holding the data
    """
    if not filename[-3:] == '.h5':
        filename += '.h5'
    data = dict()

    with hdf.File(filename, 'r') as f:
        if groups == None: #no group specified, read all
            groups = f.keys()
        for g in groups:
            # print(f[g])
            data.update({ g : dict() })
            for k in f[g].keys():
                data[g].update({ k : np.array(f[g][k]) })
    return data


def sample_surface(dshape, density, 
        mode="sample", pos=None, min_distance=0.07, seed=None):
    """
    reduce given complex data, set all other points to 0
    parameters:
        dshape        dshape, 2d tuple
        density       N (int) number of data points or loss factor (float)
        mode          'sample' to draw N samples from uniform
                      distribution (default) 
                      'grid' to grid_sample space
                      'lhs' for latin hypercube sampling
        pos           in case of 'sample', provide position vector to ensure a...
        min_distance  ..between the samples
        seed          rng seed, can be set for reproducability (default:None)
    """
    if seed:
        np.random.seed(seed)

    def get_grid(grid_shape,data_shape):
        Lx,Ly = data_shape
        Nx, Ny = grid_shape
        # floor: decrease the spacing to match grid
        divx = np.floor(Lx/Nx) 
        divy = np.floor(Ly/Ny)
        vx = np.arange(np.floor((Lx-1)%divx/2.00),Lx,divx)
        vy = np.arange(np.floor((Ly-1)%divy/2.00),Ly,divy)
        Vx,Vy = np.meshgrid(vx,vy)
        samples = (Vx+Vy*Lx).ravel().astype(int)
        if len(samples) != ((Nx+1)*(Ny+1)):
            print("spatial sampling failed...",len(samples),"instead of",(Nx+1)*(Ny+1),"points")
        return samples

    if not float(density) == 0.0:
        if mode == "grid":
            Nx = np.ceil(np.sqrt(density))-1
            Ny = np.floor(np.sqrt(density))-1
            return get_grid((Nx,Ny),dshape)
        else: # random sample
            dlen = np.prod(dshape)

            if type(density) == float:
                n_samples = int(dlen * (1 - density))
            elif type(density) == int:
                n_samples = density

            # if (mode == 'lhs'):
            # # apply latin hypercube sampling
            # from pyDOE import lhs
            # lhs(2, n_samples, 'corr', iteration)

            if np.any(pos):
                from scipy.spatial import distance

                pick = np.empty(n_samples, dtype=int)
                kk = 1
                pick[0] = np.random.choice(dlen)
                while True:
                    pick[kk] = np.random.choice(dlen)
                    if np.min(distance.pdist(pos[pick[: kk + 1], :])) >= min_distance:
                        kk += 1
                    if kk == n_samples:
                        break
            else:
                pick = np.random.choice(dlen, n_samples, replace=False).astype(int)
            return pick
    else:
        return np.arange(np.prod(dshape))


def _get_pidx(shp, patch_size):
    
    if np.sum(np.abs(np.array(shp) - np.array(patch_size))) <= 2: #patch_size == (18,18):
        patch_size = tuple([i+1 for i in patch_size])
    counter = np.arange(np.prod(shp))
    ref_idx = np.tile(np.reshape(counter, shp), (2, 2))  # should be 3x3

    prange = [shp[0] - patch_size[0] + 1, shp[1] - patch_size[1] + 1]

    pidx = np.zeros((np.prod(prange), *patch_size), dtype=int)
    for ii in range(prange[0]):
        for jj in range(prange[1]):
            pidx[ii * prange[1] + jj, :, :] = ref_idx[
                ii : ii + patch_size[0], jj : jj + patch_size[1]
            ]
    return pidx


def extract_patches_2d(A, patch_size):
    """ extracting MN subpatches of size patch_size mxn from matrix A with
    dimensions MxN, using no overlap """
    pidx = _get_pidx(A.shape[:2], patch_size)
    return A.ravel()[pidx], pidx


def reconstruct_from_patches_2d(patches, Adim, _=None , return_var = False):
    """ combining MN subpatches of size patch_size mxn to matrix A with
    dimensions Adim = (M,N), using overlap add """

    prange = (Adim[0] - patches.shape[1] + 1, Adim[1] - patches.shape[2] + 1)

    pidx = _get_pidx(Adim, patches.shape[1:])
    A = np.zeros(np.prod(Adim), dtype=patches[0].dtype)
    # mean 
    _, cidx_full = np.unique(pidx.ravel(), return_counts=True) # assume full overlap
    cidx_r = np.zeros(np.prod(Adim))
    cidx_i = np.zeros(np.prod(Adim))
    for ijdx, pid in enumerate(pidx):
        A[pid] += patches[ijdx]
        if np.linalg.norm(np.real(patches[ijdx])) > 1e-6:
            cidx_r[pid] += 1
        if np.linalg.norm(np.imag(patches[ijdx])) > 1e-6:
            cidx_i[pid] += 1
    cidx_r[cidx_r == 0] = np.inf
    cidx_i[cidx_i == 0] = np.inf
    scaling_r = 1 / cidx_r
    scaling_i = 1 / cidx_i
    A = np.real(A)*scaling_r + 1j * np.imag(A)*scaling_i

    if return_var:
        # variance
        A_var = np.zeros(np.prod(Adim), dtype=patches[0].dtype)
        cidx  = np.zeros(np.prod(Adim))
        for ijdx, pid in enumerate(pidx):
            A_var[pid] += (patches[ijdx] - A[pid])**2
            if np.linalg.norm(patches[ijdx]) > 1e-6:
                cidx[pid] += 1
        scaling = 1/cidx
        A_var *= scaling
        return np.reshape(A, Adim), np.reshape(A_var, Adim)
    else:
        return np.reshape(A, Adim), None


def calc_density(N,freq, area): # must match with soundfield
    # ## per mics circle of radius lambda/2 
    # return N / area * (np.pi/4 * (c0 / freq) ** 2)
    ## per area of wavelength**2
    return N / area * (c0/freq)**2
    # ## 1D per length of wavelength
    # return np.sqrt(N / area) * (c0 / freq)


def calc_number_of_mics(dens,freq,area):
    ## per area of wavelength**2
    return dens * area / (c0/freq)**2


class Soundfield():
    def __init__(self,
            measurement  = 'sim',
            frequency    = 1e3, dx = .05, psize = None,
            min_distance = .07,
            seed         = None,
            **kwargs):
        self.measurement  = measurement
        self.f            = frequency
        self.dx           = dx
        self.psize        = psize
        self.min_distance = min_distance
        self.pad_method   = None
        self.spatial_sampling  = 0.0
        self.loss_mode    = 'sample'
        self._seed        = seed
        self._rng         = np.random.default_rng(seed)
        self._update_flag = 0
        self.__dict__.update(kwargs)

        if not hasattr(self.f,"__iter__"):
            self.f  = np.array([float(self.f)])

        if (not self.psize) and ('patch_size_in_lambda' in kwargs):
            self.psize = self._gen_psize()

        if "1p" in self.measurement: # forcing single patch sized sound field
            self.measurement = self.measurement.replace("1p","")
            self.b_single_patch = True
        else:
            self.b_single_patch = False

        self._gen_field()

    def _gen_psize(self, freq = None):
        if not np.any(freq):
            freq = self.frequency
        return tuple( (np.ceil(c0/(np.min(freq) * self.dx)
            * self.patch_size_in_lambda+1) *np.ones(2)).astype(int))

    def _gen_field(self):
        ## import measurement data or generate

        if np.all([
            hasattr(self, '_pm'),
            hasattr(self, 'shp'),
            hasattr(self, 'psize')]):
            return

        def select_freq_idx(fdata):
            idx = np.empty(len(self.f))
            for ii,freq in enumerate(self.f):
                idx[ii] = np.abs(fdata - freq).argmin() # find closest freq
            return idx.astype(int)

        def get_position(data,prec_in_mm = 5):
            rvec = np.round(data['xyz']/prec_in_mm)*prec_in_mm/1000
            iorder = np.argsort(rvec@np.array([1e3,1,1e6]), axis=0)
            return rvec[iorder,:], iorder

        def get_measurement(data,fidx, prec_in_mm = 5):
            rm, iorder = get_position(data, prec_in_mm)
            pm = data['response'][fidx,:][:,iorder].T
            return pm, rm

        if '011' in self.measurement:
            if 'h5' in self.measurement:
                filepath = self.measurement
            else:
                filepath = "data/lab_frequency_responses.h5"
            data = read_hdf(filepath)
            fidx = select_freq_idx(data['position_0']['frequency'])

            if 'single' in self.measurement:
                self._pm, self._rm = get_measurement(data['position_0'],fidx)
                # self._tm = np.zeros(np.size(self._pm))
                # self._tm = data['position_0']['timestamp'][iorder]
                self.shp = tuple([len(np.unique(self._rm[:,0])), len(np.unique(self._rm[:,1]))])

            else: ## load arrays
                rm, iorder = get_position(data['position_0'])
                self._rm = np.empty((len(iorder),3,7))
                self._pm = np.empty((len(iorder),len(fidx),7),dtype=complex)
                # self._tm = np.empty((len(iorder),3,7))

                for iim in range(self.nof_apertures):
                    # round precision to 5 mm grid and transform to mm->m
                    self._pm[:,:,iim], self._rm[:,:,iim] = get_measurement(
                            data['position_{}'.format(iim)],fidx)
                    # self._tm[:,:,iim] = data['position_{}'.format(iim)]['timestamp']
                    self._aperturesize = self._pm[:,:,iim].shape[0]*np.ones(7)

                self.aperture_idx = 0 # default aperture for fp...

                self._tm = np.zeros(np.size(self._pm))
                self.shp = tuple([len(np.unique(self._rm[:,0,0])), len(np.unique(self._rm[:,1,0]))])
                print(np.mean(np.linalg.norm(self._pm,axis=0).ravel()))

        elif '019' in self.measurement:
            if 'h5' in self.measurement:
                filepath = self.measurement
            else:
                filepath = "data/classroom_frequency_responses.h5"
            data    = read_hdf(filepath)

            fidx = select_freq_idx( data['aperture_z1866mm']['frequency'])
            self._pm, self._rm = get_measurement(
                    data['aperture_z1866mm'], fidx, prec_in_mm = 1)
            if 'full' in self.measurement:
                self._rm[1,1] = 1.9
            self.shp = tuple([len(np.unique(self._rm[:,0])), len(np.unique(self._rm[:,1]))])

            if '3d' in self.measurement:
                pmlo, rmlo = get_measurement(data['room_019_lecture_room_P3'], fidx, prec_in_mm = 1)
                pmhi, rmhi = get_measurement(data['room_019_lecture_room_P2'], fidx, prec_in_mm = 1)
                self._aperturesize = np.array([len(self._rm), len(rmlo), len(rmhi)])
                self._rm = np.vstack((self._rm, rmlo, rmhi)) # sorted implicitly
                self._pm = np.vstack((self._pm, pmlo, pmhi))

            self._tm = np.zeros(np.size(self._pm))

        if hasattr(self, "b_single_patch"): # check for legacy compatibility
            if self.b_single_patch:
                self._get_single_patch()

        self.fdim = np.array([np.min(self.r,axis=0), np.max(self.r,axis=0)]).T
        dr_2      = np.diff(self.fdim).ravel()/(np.array([*self.shp,2])-1)/2
        self.fdimdelta = self.fdim + np.array([-dr_2, dr_2]).T

        self.areaxy = np.prod((np.max(self.r, axis=0)-np.min(self.r, axis=0))[:2])

    def _reset(self,**kwargs):
        def checkdelattr(dictionary, attribute): 
            if hasattr(dictionary, attribute): 
                delattr(dictionary, attribute)
        checkdelattr(self,'_pm')
        checkdelattr(self,'_rm')
        checkdelattr(self,'shp')
        self.__dict__.update(kwargs)
        self._gen_field()

    def _get_single_patch(self):
        """ selecting single patch from """
        # 0 select patch
        if not hasattr(self,"patch_number"):
            self.patch_number = self._rng.choice(self.pidx.shape[0])
            print("random patch number:",self.patch_number)
        self._pidx = self.pidx[self.patch_number,:,:]
        idx        = self.pidx.ravel()

        # 1 store pm, rm, t in primary variable fields
        self.pm    = self._pm[idx]
        # self.um    = self._um[:,idx]
        self.rm    = self._rm.reshape((-1,3))[idx]
        if hasattr(self, '_tm'):
            self.tm    = self._tm[idx]

        # update pidx to range, shp == psize
        self.pad_method = None
        self.shp   = self.psize
        delattr(self, '_pidx')
        _ = self.pidx

    def update(self, **kwargs):
        self.__dict__.update(**kwargs)
        self._update_flag = 1

    @property
    def fmask(self):
        fmask = np.zeros(self.fp.shape[0], dtype=self.fp.dtype)
        fmask[self.sidx] = 1
        return fmask

    @property
    def mask(self):
        return self.fmask.reshape(self.shp)

    @property
    def mic_density(self): 
        return calc_density(self.N, self.f, self.areaxy)

    @property
    def avg_distance(self): 
        from scipy.spatial import Delaunay
        from itertools import combinations
        points = self.r[self.sidx][:,:2]
        if len(points) <= 1: return 0
        if len(points) == 2: return np.linalg.norm(points[0]-points[1])
        # calc delaunay triangulation
        tri = Delaunay(points)

        # avg length of all unique edges
        triangles = tri.simplices
        # get all the unique edges 
        all_edges = set([tuple(sorted(edge)) for item in triangles for edge in combinations(item,2)])
        # compute and return the average dist 
        return np.mean([np.linalg.norm(points[edge[0]]-points[edge[1]]) for edge in all_edges])

    @property
    def N(self): 
        _ = self.sidx
        return self.spatial_sampling

    @property
    def prms2(self):
        # mean p_rms^2 [Pa^2]
        return np.mean(np.abs(self.fp)**2)

    @property
    def sprms2(self):
        # mean p_rms^2 at sampling points [Pa^2]
        return np.mean(np.abs(self.fp[self.sidx])**2)

    @property
    def wavelength(self):
        return c0/self.frequency

    @property
    def extent(self):
        """return (xmin,xmax,ymin,ymax)"""
        dims =[[np.min(self.r[:,1]), np.max(self.r[:,1])],
               [np.min(self.r[:,0]), np.max(self.r[:,0])],
               [np.min(self.r[:,2]), np.max(self.r[:,2])]]
        aperture = np.array(dims).ravel()[:4]
        dx_2     = np.diff(aperture[:2])/(self.shp[0]-1)/2
        dy_2     = np.diff(aperture[2:])/(self.shp[1]-1)/2
        return tuple((aperture + np.array([-dx_2, dx_2, -dy_2, dy_2]).T)[0])


    def _cat_patches(self, x):
        # padding only applied for single aperture case!
        def ifpad(x):
            if self.pad_method:
                if 'zero' in self.pad_method:
                    mode = "constant"
                elif 'reflect' in self.pad_method:
                    mode = "reflect"
                x = np.pad(x, self.psize[0]-1, mode = mode)
            return x

        if np.ndim(x) == 1: 
            # single aperture, single frequency
            x = np.reshape(x, self.shp)
            self.pidx; # init patching
            x = ifpad(x)
            return x.ravel()[self.pidx]
        elif np.ndim(x) == 2: 
            xs = x.shape[1:]
            x = ifpad(x.reshape(self.shp)).reshape((-1,*xs))
            if x.shape[1] == len(self.f):
                # single aperture, multiple frequencies
                x_patched = np.array([pp[self.pidx] for pp in x.T])
                x_patched = np.moveaxis(x_patched,0,-1)
                return x_patched
            else: 
                # multiple apertures, single frequency
                x_patched =  np.array([pp[self.pidx] for pp in x])
                return x_patched.reshape((-1, *self.psize))
        else: 
            #multiple apertures, multiple frequencies
            x_patched = np.array([pp[:,self.pidx] for pp in x.T])
            x_patched = np.moveaxis(x_patched,1,-1)
            return x_patched.reshape((-1, *self.psize, len(self.f)))

    @property
    def fsp(self): # sampled versions ONLY take the first aperture
        return self.fmask[:,np.newaxis] * self.fp

    @property
    def fp(self): # bypass self.fp to include all apertures (for learning)
        if hasattr(self,'pm'):
            pm = self.pm
        elif hasattr(self,'_pm'):
            pm = self._pm
        if np.ndim(pm) <= 2:
            return pm
        else:
            return pm[:,:, self.aperture_idx]

    @property
    def sp(self):
        # return self.fsp.reshape(self.shp + (self.fp.shape[-1],)) #MULTIFREQ
        return self.p*self.mask

    @property
    def patches(self):
        if hasattr(self,"pm"):
            fp = self.pm
        else:
            fp = self._pm
        return self._cat_patches(fp).squeeze()

    @property
    def u(self):
        if hasattr(self,"um"):
            fu = self.um
            return fu.reshape(-1,*self.shp)
        elif hasattr(self,"_um"):
            fu = self._um
            return fu.reshape(-1,*self.shp)
        else:
            return np.zeros((3,*self.shp))

    @property
    def IJ(self): # active + reactive intensity
        return self.p[np.newaxis,...]*self.u.conj()/2

    @property
    def spatches(self):
        return self._cat_patches(self.fsp)

    @property
    def fspatchesall(self):
        fmask = np.zeros(self._pm.shape, dtype=self.fp.dtype)
        fmask[self.sidx,:] = 1
        pm = self._pm * fmask
        spatchesall = self._cat_patches(pm)
        fspatchesall = spatchesall.reshape((-1, np.prod(self.psize), len(self.f)))
        return fspatchesall

    @property
    def fpatches(self):
        return self.patches.reshape((-1, np.prod(self.psize), len(self.f)))

    @property
    def fspatches(self):
        return self.spatches.reshape((-1, np.prod(self.psize), len(self.f)))

    @property
    def fspatches_nz(self):
        # returns list of tuples with (measurements, patch_indices) for each patch
        return [(fspatch.compress(fspatch!=0)[:,np.newaxis], np.where(fspatch!=0)[0] )for fspatch in self.fspatches.squeeze()]

    @property
    def p(self):       
        fpshape = self.fp.shape
        if np.prod(self.shp) == fpshape[-1]:
            p = self.fp.reshape(self.shp)
        elif np.prod(self.shp) == fpshape[0]:
            p = self.fp.reshape(tuple(self.shp) + (self.fp.shape[-1],)) # MUTLIFREQ
        return p.squeeze()

    @property
    def padlen(self):
        return self.psize[0]-1 if self.pad_method is not None else 0

    @property
    def paddedshape(self):
        return (self.shp[0]+2*self.padlen, self.shp[1]+2*self.padlen)

    @property
    def pidx(self): # returns patches
        if not hasattr(self,"_pidx"):
            x = self.p
            if self.pad_method is not None:
                x = np.pad(x, self.padlen, mode = "constant")
            _, self._pidx = extract_patches_2d(x, self.psize)
        return self._pidx

    def multifreq_patches(self, target_freq = None):
        if target_freq is None:
            target_freq = self.frequency
        target_psize = self._gen_psize(freq = target_freq)

        mp = list()
        for ii,ff in enumerate(self.f): #loop through frequencies
            ffpsize = self._gen_psize(freq = ff)
            # TODO: Scale before patch extraction for fewer interpolation artifacts
            fp = np.vstack([ 
                extract_patches_2d(self._pm[:,ii,nn].reshape(self.shp), ffpsize)[0] 
                for nn in range(self.nof_apertures)])
            fp = fp.reshape((fp.shape[0],-1))
            fp = scale_patches(fp, self.dx, ff, self.dx, target_freq,
                    self.patch_size_in_lambda,
                    # mode='direct', # not for DL paper
                    )
            mp.append(fp) # scale patches and append

        mp = np.vstack(mp)
        return mp, target_psize

    @property
    def nof_measurements(self):
        return len(self.sidx)

    @property
    def sidx(self):
        if (self.spatial_sampling != 0.0):
            if not hasattr(self,"sample_idx"):
                # print(" > sampling aperture", end = " ")
                from time import time
                tic = time()
                # convert density to absolute number of mics
                if (self.spatial_sampling < 0.0):
                    self.spatial_sampling = int(np.round( 
                        calc_number_of_mics( -self.spatial_sampling, self.f,
                            self.areaxy)))

                # see if min_distance can be kept
                if   self.spatial_sampling / np.prod(self.shp) > .4:
                    self.min_distance = 1e-5
                elif self.spatial_sampling / np.prod(self.shp) > .09:
                    self.min_distance = self.dx + 1e-5
                if not hasattr(self,'_seed'):
                    self._seed = None
                self.sample_idx = sample_surface(self.shp,
                        self.spatial_sampling,
                        self.loss_mode,
                        pos = self.r,
                        min_distance = self.min_distance,
                        seed = self._seed)
                self.spatial_sampling = len(self.sample_idx)
                # print("{:_>26}".format(" in {:.2f}".format(time()-tic)))
            return self.sample_idx
        else:
            return np.arange(self.fp.shape[0])

    @property
    def frequency(self):
        if hasattr(self.f, '__iter__'):
            return self.f[0]
        else:
            return self.f

    @property
    def k(self):
        return 2*np.pi*self.frequency/c0

    @property
    def r(self):
        if hasattr(self,'rm'):
            rm = self.rm
        else:
            rm = self._rm

        if np.ndim(rm) <= 2:
            return rm
        else:
            return rm[:,:,0]

    @property
    def t(self):     
        if hasattr(self,"tm"):
            return self.tm
        elif hasattr(self,"_tm"):
            return self._tm
        else:
            print("time not available")


def scale_patches(patches, dx_current, f_current, dx_target, f_target,
        patch_size_in_lambda, mode='spline'):
    # args: soundfield, patches (nof_features x nof_patches), f_current, f_target
    # TODO: use direct interpolation, as in SoundfieldReconstruction
    side_current = int(np.sqrt(patches.shape[1]))
    wl_current = c0/f_current
    current    = np.arange(float(side_current))
    current   *= dx_current/wl_current # scale in lambda
    wl_target  = c0/f_target
    target     = np.arange(np.ceil(wl_target/dx_target * patch_size_in_lambda)+1)
    target    *= dx_target/wl_target

    if mode == 'direct': # not tested
        scale = 1/np.max(np.diff(current))
        tmp = np.zeros((len(current),len(target)),dtype=patches.dtype)
        pr = np.zeros((patches.shape[0],len(target),len(target)),dtype=patches.dtype)
        for kk,pm in enumerate(patches.reshape((-1,side_current,side_current))):
            for ii,xi in enumerate(current):
                for jj,yj in enumerate(target):
                    tmp[ii,jj] = np.inner(pm[ii,:],np.sinc((yj-current)*scale))
            for jj,yj in enumerate(target):
                for ii,xi in enumerate(target):
                    pr[kk,ii,jj] = np.inner(tmp[:,jj],np.sinc((xi-current)*scale))

        return pr.reshape((patches.shape[0],-1))
    else:
        from scipy import interpolate

        nof_patches = patches.shape[0]
        p_scaled = np.empty((nof_patches, int(len(target)**2) ), dtype=patches.dtype)
        for ii in range(nof_patches):

            fr = interpolate.interp2d( current, current, np.real(patches[ii]), kind='quintic')
            p_scaled[ii] = fr(target,target).ravel('F')

            if patches.dtype == complex:
                fi = interpolate.interp2d( current, current, np.imag(patches[ii]), kind='quintic')
                p_scaled[ii] += 1j*fi(target,target).ravel('F')
        return p_scaled


def default_soundfield_config(measurement, **kwargs):
    config = dict({
            'frequency'   : c0,
            'min_distance': 0.07,
            'measurement' : measurement,
            'pad_method'  : None,
            'patch_size_in_lambda' : 1.0,
            })
    [config.update({key:value}) for key,value in kwargs.items() if key in config.keys()]
    if '011' in measurement:
        config.update({
            'frequency' : 600,
            'dx'   : .05,
            'rdim' : [[0,4.41],[0,3.31],[0,2.97]],
            'nof_apertures' : 7,
            })
    elif '019' in measurement:
        config.update({
            'frequency' : 600,
            'dx'   : .025,
            'rdim' : [[0,9.45],[0,6.63],[0,2.97]],
            })

    config.update(kwargs)
    return config