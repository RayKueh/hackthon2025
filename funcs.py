from scipy.interpolate import griddata
import numpy as np
from scipy.ndimage import gaussian_filter #從 scipy.ndimage 導入 gaussian_filter
from matplotlib import pyplot as plt
import pandas as pd
import netCDF4 as nc
import h5py
from numba import jit,njit,prange
from numpy import linalg
from time import sleep
from tqdm import tqdm, trange
import torch, torch.linalg
from tqdm import tqdm
import shutil
from time import gmtime, strftime
from scipy.fft import *
import matplotlib
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal
import xarray as xr
from scipy.interpolate import *
import scipy.linalg as lg 

from scipy.spatial import distance_matrix

############################################################


# lon_name: str 
def weighted_mean(target,lon_name=None):
    try: 
        weights = np.cos(np.deg2rad(target['lat']))
    except: 
        weights = np.cos(np.deg2rad(target[lon_name]))

    weights.name = "weights"
    
    temp = target.weighted(weights) 
    output = temp.mean('lat')
    return output 

# x: time,lon (meaned lat axis) 
def smooth_x(x,cut_off,dx): 
    sig_fft = fft(x)
    sig_fft_filtered = sig_fft.copy()
    freq = fftfreq(len(x), dx)
    
    sig_fft_filtered[np.abs(freq) > cut_off] = 0
    filtered = ifft(sig_fft_filtered)
    return filtered 

# x: time,lon (meaned lat axis)
def smooth_t(x,c_min,c_max,dt):
    sig_fft = fft(x)
    sig_fft_filtered = sig_fft.copy()
    freq = fftfreq(len(x), dt)
    
    sig_fft_filtered[np.abs(freq) > c_max] = 0
    sig_fft_filtered[np.abs(freq) < c_min] = 0
    
    filtered = ifft(sig_fft_filtered)
    return filtered

############### Plot MJO #################################
def plot_mjo_phase_space_ax(ax):
    import numpy as np
    import matplotlib.lines as lines

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xticks(range(-4, 5))
    ax.set_yticks(range(-4, 5))

    # plot MJO phase diagram lines
    ax.add_line(lines.Line2D([np.cos(np.pi/4), 4], [np.sin(np.pi/4), 4], color='black', linestyle='--', lw=1))
    ax.add_line(lines.Line2D([np.cos(3*np.pi/4), -4], [np.sin(np.pi/4), 4], color='black', linestyle='--', lw=1))
    ax.add_line(lines.Line2D([np.cos(np.pi/4), 4], [np.sin(7*np.pi/4), -4], color='black', linestyle='--', lw=1))
    ax.add_line(lines.Line2D([np.cos(3*np.pi/4), -4], [np.sin(7*np.pi/4), -4], color='black', linestyle='--', lw=1))
    ax.add_line(lines.Line2D([-4, -1], [0, 0], color='black', linestyle='--', lw=1))
    ax.add_line(lines.Line2D([1, 4], [0, 0], color='black', linestyle='--', lw=1))
    ax.add_line(lines.Line2D([0, 0], [1, 4], color='black', linestyle='--', lw=1))
    ax.add_line(lines.Line2D([0, 0], [-1, -4], color='black', linestyle='--', lw=1))

    amp1_circ = plt.Circle((0, 0), 1, color='black', fill=False)
    ax.add_patch(amp1_circ)

    # add phase diagram texts
    ax.text(1, 3, 'Phase 6', size='medium', weight='semibold')
    ax.text(-2, 3, 'Phase 7', size='medium', weight='semibold')
    ax.text(2.8, 1, 'Phase 5', size='medium', weight='semibold', ha='center')
    ax.text(-2.8, 1, 'Phase 8', size='medium', weight='semibold', ha='center')
    ax.text(1, -3, 'Phase 3', size='medium', weight='semibold')
    ax.text(-2, -3, 'Phase 2', size='medium', weight='semibold')
    ax.text(2.8, -1, 'Phase 4', size='medium', weight='semibold', ha='center')
    ax.text(-2.8, -1, 'Phase 1', size='medium', weight='semibold', ha='center')

    ax.text(0, 3.7, 'Pacific Ocean', ha='center', bbox=dict(facecolor='white', edgecolor='white'))
    ax.text(0, -3.8, 'Indian Ocean', ha='center', bbox=dict(facecolor='white', edgecolor='white'))
    ax.text(-3.8, 0, 'West. Hem., Africa', va='center', rotation='vertical', bbox=dict(facecolor='white', edgecolor='white'))
    ax.text(3.7, 0, 'Maritime Continent', va='center', rotation='vertical', bbox=dict(facecolor='white', edgecolor='white'))

    ax.set_xlabel('RMM1')
    ax.set_ylabel('RMM2')
    ax.grid(True)

def plot_kw_phase_space_ax(ax,amp=300,std=100,scale=80,cle_color = 'orange'):
    import numpy as np
    import matplotlib.lines as lines

    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_xticks(range(-300, 301,100))
    ax.set_yticks(range(-300, 301,100)) 

    
    # plot MJO phase diagram lines
    ax.add_line(lines.Line2D([np.cos(np.pi/4), amp], [np.sin(np.pi/4), amp], color='black', linestyle='--', lw=1))
    ax.add_line(lines.Line2D([np.cos(3*np.pi/4), -amp], [np.sin(np.pi/4), amp], color='black', linestyle='--', lw=1))
    ax.add_line(lines.Line2D([np.cos(np.pi/4), amp], [np.sin(7*np.pi/4), -amp], color='black', linestyle='--', lw=1))
    ax.add_line(lines.Line2D([np.cos(3*np.pi/4), -amp], [np.sin(7*np.pi/4), -amp], color='black', linestyle='--', lw=1))
    ax.add_line(lines.Line2D([-amp, -std], [0, 0], color='black', linestyle='--', lw=1))
    ax.add_line(lines.Line2D([std, amp], [0, 0], color='black', linestyle='--', lw=1))
    ax.add_line(lines.Line2D([0, 0], [std, amp], color='black', linestyle='--', lw=1))
    ax.add_line(lines.Line2D([0, 0], [-std, -amp], color='black', linestyle='--', lw=1))

    
    amp1_circ = plt.Circle((0, 0), std, color=cle_color,alpha=0.3, fill=True)
    ax.add_patch(amp1_circ)

    # add phase diagram texts

    
    ax.text(1   *scale, 3*scale, 'Phase 2', size='medium', weight='semibold')
    ax.text(-2  *scale, 3*scale, 'Phase 3', size='medium', weight='semibold')
    ax.text(2.8 *scale, 1*scale, 'Phase 1', size='medium', weight='semibold', ha='center')
    ax.text(-2.8*scale, 1*scale, 'Phase 4', size='medium', weight='semibold', ha='center')
    ax.text(1   *scale, -3*scale, 'Phase 7', size='medium', weight='semibold')
    ax.text(-2  *scale, -3*scale, 'Phase 6', size='medium', weight='semibold')
    ax.text(2.8 *scale, -1*scale, 'Phase 8', size='medium', weight='semibold', ha='center')
    ax.text(-2.8*scale, -1*scale, 'Phase 5', size='medium', weight='semibold', ha='center')

    '''
    ax.text(0, 3.7, 'Pacific Ocean', ha='center', bbox=dict(facecolor='white', edgecolor='white'))
    ax.text(0, -3.8, 'Indian Ocean', ha='center', bbox=dict(facecolor='white', edgecolor='white'))
    ax.text(-3.8, 0, 'West. Hem., Africa', va='center', rotation='vertical', bbox=dict(facecolor='white', edgecolor='white'))
    ax.text(3.7, 0, 'Maritime Continent', va='center', rotation='vertical', bbox=dict(facecolor='white', edgecolor='white'))

    ax.set_xlabel('RMM1')
    ax.set_ylabel('RMM2')
    '''
    ax.grid(True)



############## EOF analysis################################ 
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.signal import filtfilt, butter
from EOF import EOF

class MJO_Analyzer: 
    def __init__(self,data,varnames,dt,latname='lat'):
    ## data: xarray dataset, var_names: variables we concern. 
        self.data     = data      # xarray , dim: (time,lon,lat) 
        self.varnames = varnames  # list of str 
        self.latname  = latname   # str of key of latitude 
        self.dt       = dt   # time steps. Unit: day 
        
    def get_EOF_in_one(self):
        self._load_data()
        self._slice_range(-15,15) 
        self._lat_wt_ave() 
        self.xr_bandpass_t()
        self.preprocess_for_EOF()
        self._combine_and_compute_eof()

    def get_EOF_in_one_2d(self):
        self._load_data()
        self._slice_range(-15,15) 
        
        self.xr_bandpass_t()
        self.preprocess_for_EOF()
        self._combine_and_compute_eof()

    
        
    def _load_data(self): 
        self.extracted_data = self.data[self.varnames].copy()
        
    
    ## min,max:the limit of slice 
    ## lat_name: the name of latitute. it can be any other variavles  
    def _slice_range(self,min,max):
        lat = self.latname
        self.extracted_data = self.extracted_data.sel(lat=slice(min, max))

    def _lat_wt_ave(self):
        lat = self.extracted_data[self.latname] 
        weights = np.sqrt(np.cos(np.deg2rad(lat)))
        weights /= weights.sum()

        temp = self.extracted_data 
        temp_weighted = temp.weighted(weights)
        print(temp_weighted) 

        self.exct_wt_mean = temp_weighted.mean((self.latname))
        
    
    def butter_bandpass_filter(self,data, axis ,lowcut, highcut):
        # axis = 0 for t 
        nyquist = 0.5 * self.dt
        b, a = butter(1, [lowcut / nyquist, highcut / nyquist], btype='band')
        return filtfilt(b, a, data, axis=axis)
    
    def xr_bandpass_t(self, dim='time', lowcut=1/90, highcut=1/30):
        self.bandpassed_t = self.exct_wt_mean.copy()
    
        for i, key in enumerate(self.varnames): 
            temp = self.exct_wt_mean[key]
            filtered = self.butter_bandpass_filter(temp.to_numpy(), axis=0,lowcut=lowcut, highcut=highcut)
            self.bandpassed_t[key] = xr.DataArray(filtered, dims=temp.dims, coords=temp.coords)

            

    def preprocess_for_EOF(self): 
        mean = self.bandpassed_t.mean(dim='time')
        std  = self.bandpassed_t.std(dim='time')
        # z-score all fields (broadcasting rules take care of dims)
        self.bandpassed_t_sted = (self.bandpassed_t - mean) / std
        self.bandpassed_t_sted = self.bandpassed_t_sted.interpolate_na(dim='time', method="linear")

    def _combine_and_compute_eof(self):
        temp = {}
        temp_for_dim = self.bandpassed_t_sted[self.varnames[0]].squeeze().to_numpy() 
        print(temp_for_dim.shape)

         
        time_dim, lon_dim = temp_for_dim.shape

        
        
        for i,key in enumerate(self.varnames): 
            
            np_arr_now = self.bandpassed_t_sted[key].to_numpy()
            np_arr_now = np_arr_now.reshape((time_dim,lon_dim))
            
            temp[key] = np_arr_now
        
        combined = np.concatenate(list(temp.values()), axis=1)
         
        eof_instance = EOF((combined,), n_components=10, field="1D")
        eof_instance.get()
        
        self.eof_instance = eof_instance 
        
        PC1 = eof_instance.PC[0]
        PC2 = eof_instance.PC[1]

        # Normalize PCs
        self.PC1 = (PC1 - PC1.mean()) / PC1.std()
        self.PC2 = (PC2 - PC2.mean()) / PC2.std()
        self.explained = eof_instance.explained[:3]
        
    def get_pc1_pc2(self):
        return self.PC1, self.PC2

    def get_pcs_and_eof(self): 
        # Extract EOFs and PCs
        # EOF1, EOF2, EOF3 = eof_instance.EOF[:3]
        # PC1, PC2, PC3 = eof_instance.PC[:3]
        return self.eof_instance 
        # don't forget to normalize if you want to further discus
        
        
    

    
    #----SVD - based EOF (POD) ----------------------------------------# 

    def preprocess_np_array(self,data): # data: (time,*) np.array  
        
        anomaly = data - np.mean(data, axis=0)
        return anomaly / np.std(anomaly, axis=0)

                                
    def compute_eof_svd_in_one(self,mode,time_steps): 
        self._load_data()

        data_desired  = {} 
        U_desired   = {} 
        s_desired   = {} 
        vh_desired  = {} 
        
        
        for i,key in enumerate(self.varnames): 
            data_now = self.extracted_data[key].to_numpy() 
            data_now_bdpass_t = self.butter_bandpass_filter(data_now, axis=0 ,lowcut=1/90, highcut=1/30)
            
            aa,bb,cc = np.shape(data_now_bdpass_t) 
            data_now_bdpass_t = data_now_bdpass_t.reshape((aa,bb*cc))
            data_now_bdpass_t = self.preprocess_np_array(data_now_bdpass_t)
            
            U,s,vh = lg.svd(data_now_bdpass_t[:time_steps].T,full_matrices=False)
            U_40 = U[:,:mode]
            s_40 = s[:mode]
            s_40_1 = np.diag(1/s_40)

            Q = np.matmul(s_40_1,U_40.T)
            print(Q.shape)
            print(data_now_bdpass_t.shape)
            vh_data = np.matmul(Q,data_now_bdpass_t.T).T

            vh_desired[key] = vh_data 
        return vh_desired

    def compute_eof_svd_in_one_ai(self, mode, time_steps):
        self._load_data()
    
        PC_result = {}
    
        for i, key in enumerate(self.varnames):
            data_now = self.extracted_data[key].to_numpy()
    
            # Apply bandpass filter in time
            data_now_bdpass_t = self.butter_bandpass_filter(data_now, axis=0, lowcut=1/90, highcut=1/30)
    
            # Reshape to (time, space)
            nt, ny, nx = data_now_bdpass_t.shape
            data_flat = data_now_bdpass_t.reshape(nt, ny * nx)

            data_flat = (data_flat - np.mean(data_flat, axis=0)) / np.std(data_flat, axis=0)
    
            # Slice to first `time_steps` time points
            data_flat = data_flat[:time_steps, :]
    
            # Standardize: remove mean and divide by std
            mean = np.mean(data_flat, axis=0)
            std = np.std(data_flat, axis=0)
            standardized_data = (data_flat - mean) / std
    
            # Perform SVD
            U, s, Vt = np.linalg.svd(standardized_data, full_matrices=False)
    
            # Keep leading modes
            PC = np.dot(standardized_data, Vt.T[:, :mode])  # shape: (time_steps, mode)
    
            # Store result
            PC_result[key] = PC
    
        return PC_result
        #----- SVD based EOF-----End------------------#                 

        ############# Wheeler and klaudis ###################
      
import sys;
import numpy as np;
import xarray as xr;

from matplotlib import pyplot as plt;
from scipy.ndimage import convolve1d;



class Wheeler_and_klaudis(MJO_Analyzer):
    def __init__(self, data, varnames, dt, latname='lat', new_param=None):
        super().__init__(data, varnames, dt, latname)
        self._load_data() 
    
    def _slice_range(self,min,max): 
        super()._slice_range(min = min, max = max)
        # Gives new attr: self.extracted_data 

    def processing_data(self): 
        # remove climatology and zonal mean
        self.data_ano: dict[str, np.ndarray] = {
                        exp: data[exp] - data[exp].mean(dim={"time", "lon"})
                        for exp in data.keys()
                        };
        
        # symmetrize and asymmetrize data
        self.data_sym: dict[str, np.ndarray] = {
                        exp: (data_ano[exp] + data_ano[exp].isel(lat=slice(None, None, -1))) / 2.0
                        for exp in data_ano.keys()
                        };
        
        self.data_asy: dict[str, np.ndarray] = {
                        exp: (data_ano[exp] - data_ano[exp].isel(lat=slice(None, None, -1))) / 2.0
                        for exp in data_ano.keys()
                        };
        
        # windowing data
        self.hanning: np.ndarray = np.hanning(120)[:, None, None, None];
        
        self.sym_window: dict[str, np.ndarray] = {
                        exp: np.array([
                            data_sym[exp].isel(time=slice(i*60, i*60+120)) * hanning
                            for i in range(5)
                                    ])
                        for exp in data_sym.keys()
                        };
        
        self.asy_window: dict[str, np.ndarray] = {
                        exp: np.array([
                        data_asy[exp].isel(time=slice(i*60, i*60+120)) * hanning
                        for i in range(5)
                        ])
                        for exp in data_asy.keys()
                        };


    






######################################
######################################
######################################
######################################
######################################















import numpy as np
from EOF import EOF

class MJO_Analyzer_2D:
    def __init__(self, data_dict, n_modes):
        """
        data_dict: dictionary with variable name -> np.array of shape (time, lat, lon)
        n_modes: number of EOF modes to extract
        """
        self.data_dict = data_dict
        self.n_modes = n_modes
        self.eof_instance = None
        self.PCs = {}
        self.EOFs = {}

    def preprocess(self):
        # Standardize each variable in time
        for key in self.data_dict:
            data = self.data_dict[key]  # shape: (time, lat, lon)
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            self.data_dict[key] = (data - mean) / std

    def compute_eof(self):
        self.preprocess()
        arr_tuple = tuple(self.data_dict[key] for key in self.data_dict)
        self.eof_instance = EOF(arr_tuple, n_components=self.n_modes, field="2D")
        self.eof_instance.get()
        self.EOFs = self.eof_instance.EOF  # shape: (n_modes, space)
        for i in range(self.n_modes):
            self.PCs[i] = self.eof_instance.PC[i]  # shape: (time,)

    def get_pc(self, mode):
        return self.PCs.get(mode, None)

    def get_explained(self):
        return self.eof_instance.explained if self.eof_instance else None



























