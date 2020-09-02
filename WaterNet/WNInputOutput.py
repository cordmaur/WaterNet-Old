from pathlib import Path
import numpy as np
import gdal
import math
import torch
from torch.utils import data
import time
import pickle
from PIL import Image as PilImg
# import WNFastaiClasses

import lightgbm as lgb
# import pdb

# from fastai.vision import *
# from fastai.metrics import error_rate
#
# from PIL import Image as PilImg

import matplotlib.pyplot as plt


def search_file(paths: list, name, recursive=False):
    pattern = '**/*' if recursive else '*'

    for path in paths:
        file = [file for file in path.glob(pattern) if name in file.name]

        if len(file) > 0:
            return file[0]

    print(f'File {name} not found in {paths}')

    return None


def open_tif_file(paths: list, key, name, recursive=False):
    file = search_file(paths, name, recursive)

    ds = gdal.Open(file.as_posix())

    if ds:
        print('Opening file: ' + file.name)

        #         set_trace()

        array = ds.ReadAsArray(buf_xsize=10980, buf_ysize=10980, resample_alg=gdal.GRA_Average)
        array = array / 10000 if key[-1] != 'i' else array
        return {key: array}

    else:
        print('File {} not found'.format(file.name))
        return {}


def open_gdal_ds(paths: list, key, name, recursive=False):
    file = search_file(paths, name, recursive)

    if file is not None:
        ds = gdal.Open(file.as_posix())
        if ds is not None:
            print('Opening as dataset: ' + file.name)
            return {key: ds}
        else:
            print(f'Dataset for band {key} not found in {name}')

    return {}


def array2raster(filename, array, geo_transform, projection, nodatavalue=0, dtype=gdal.GDT_Float32):

    cols = array.shape[1]
    rows = array.shape[0]

    print('Saving image: ' + filename)

    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(filename, cols, rows, 1, dtype, options=['COMPRESS=PACKBITS'])
    out_raster.SetGeoTransform(geo_transform)
    out_raster.SetProjection(projection)
    outband = out_raster.GetRasterBand(1)
    outband.SetNoDataValue(nodatavalue)
    outband.WriteArray(array)
    outband.FlushCache()
    return


def save_obj(path, obj):
    with open(str(path), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(str(path), 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def rename_files(path, old_str, new_str):
    [f.rename(f.with_name(f.name.replace(old_str, new_str))) for f in path.iterdir() if old_str in f.name]


def normalize_band(img, band):
    return (img[band] + 1) / 2


def remove_negative(img, band):
    b = img[band].copy()
    b[b == -1.] = np.min(b[b != -1.])
    if np.min(b) < 0:
        b = b - np.min(b)
    return b


def create_training_patches_old(train_imgs, out_path, channels_first=True, ext='npy'):
    for key, value in train_imgs.items():
        # process the image
        img = WNSatImage(value['img'])

        img.band_math('n_ndwi', lambda x: normalize_band(x, 'ndwi'))
        img.band_math('n_mndwi', lambda x: normalize_band(x, 'mndwi'))
        img.band_math('n_B11', lambda x: remove_negative(img, 'B11'))

        pproc = WNPatchProcessor(img)

        pproc.create_patches(['n_mndwi', 'n_ndwi', 'n_B11', 'B2'], 366, 183, channels_first=channels_first)
        pproc.save_patches(out_path / 'images', key, ext=ext)

        # process the water mask (Label)
        lbl = WNImage(value['lbl'])
        lbl.band_math('water_mask', lambda x: np.where(x[0] == 255, 0, x[0]))

        lproc = WNPatchProcessor(lbl)
        lproc.create_patches('water_mask', 366, 183, channels_first=channels_first)
        lproc.save_patches(out_path / 'labels', key, ext=ext)

        lproc.clear()
        pproc.clear()


def create_custom_patches(img, bands, size, shift, bands_math={}, chnls_first=True):
    # function
    for key, value in bands_math.items():
        img.band_math(key, value)

    pproc = WNPatchProcessor(img)

    pproc.create_patches(bands+list(bands_math.keys()), size, shift, chnls_first)

    return pproc


def create_train_patches(img, lbl, out_path, size, shift, bands, bands_math={}, chnls_first=True, ext='npy',
                         base_name='', proc_label={}, fill_nan=None):

    out_path = Path(out_path)

    for i, path_name, maths in zip([img, lbl], ['images', 'labels'], [bands_math, proc_label]):
        if i is not None:
            path = out_path / path_name

            bands = bands if path_name == 'images' else [0]

            img_proc = create_custom_patches(i, bands, size, shift, maths, chnls_first=chnls_first)

            # else:
            #     if len(proc_label) == 0:
            #         img_proc = create_custom_patches(i, 0, size, shift)
            #     else:
            #         img_proc = create_custom_patches(i, list(proc_label.keys())[0], size, shift, bands_math=proc_label)

            img_proc.save_patches(path, base_name, ext, fill_nan=fill_nan)
            img_proc.clear()

    return 'Processing completed'


def auto_train_patches_creation(imgs_dict, out_path, bands, size, shift, bands_math={}, proc_label={},
                                shape=(10980, 10980)):
    for key, value in imgs_dict.items():
        print(f'Creating patches for {key}')

        if 'img' in value:
            img = WNSatImage(value['img'])
            img.shape = shape
        else:
            img = None

        if 'lbl' in value:
            lbl = WNImage(value['lbl'])
            lbl.shape = shape
        else:
            lbl = None

        create_train_patches(
            img,
            lbl,
            out_path=out_path,
            size=size,
            shift=shift,
            bands=bands,
            bands_math=bands_math,
            chnls_first=True,
            base_name=key,
            proc_label=proc_label
        )


def predict_patches(proc, learn):
    lst_img = []
    lst_mask = []

    for idx in range(len(proc)):
        ms_img = WNFastaiClasses.MSImage(torch.tensor(proc[idx]).float())
        pred_img, mask, probs = learn.predict(ms_img)

        # array = np.array(mask[0].data).squeeze()[:,:,np.newaxis]
        array = np.array(mask[0].data).squeeze()

        lst_img.append(pred_img)
        lst_mask.append(mask.squeeze().numpy())
        lst_probs.append(probs.numpy())

    return lst_img, lst_mask, lst_probs


def predict_image(img, learn, bands, size, shift, bands_math={}):
    pproc = create_custom_patches(img, bands, size, shift, bands_math)

    _, masks, _ = predict_patches(pproc, learn)
    out_proc = WNPatchProcessor(from_patches=masks)
    out_proc.set_format(0, 366, 366, True, img.geo_transform, img.projection)

    return out_proc

####################################################################################
class WNImage:

    resampling = gdal.GRA_NearestNeighbour
    # resampling = gdal.GRA_Average

    def __init__(self, path=None, shape=None):

        self.path_, self.shape_ = path, shape

        self.dataset = gdal.Open(str(path)) if path is not None else None
        self.loaded_bands_ = {}
        self.calc_bands_ = {}

    # @staticmethod
    def normalized_difference(self, b1, b2, name=None):
        if name is None:
            min_cte = np.min([self[b1], self[b2]])
            if min_cte <= 0:
                min_cte = -min_cte + 0.0001
            else:
                min_cte = 0

            return ((self[b1]+min_cte) - (self[b2]+min_cte)) / ((self[b1]+min_cte) + (self[b2]+min_cte))
        else:
            return self.band_math(name, lambda x: self.normalized_difference(b1, b2, None))

    @staticmethod
    def create_nan_mask(img):
        nan_mask = np.zeros(img.shape).astype('bool')
        for band in img.available_bands:
            nan_mask |= np.isnan(img[band])
        return nan_mask.astype('float32')

    @property
    def calc_bands(self):
        return list(self.calc_bands_.keys())

    @property
    def available_bands(self):
        # available bands is a combination of inner bands and calculated bands
        if self.dataset is not None:
            return list(range(self.dataset.RasterCount)) + self.calc_bands
        else:
            return []

    @property
    def data_source(self):

        if self.dataset is None:
            print(f'No dataset in WNImage')
            return None
        else:
            return self.dataset

    @property
    def projection(self):
        return self.data_source.GetProjection()

    @property
    def geo_transform(self):
        return self.data_source.GetGeoTransform()

    @property
    def path(self):
        if self.dataset is not None:
            return Path(self.dataset.GetFileList()[0])
        else:
            return self.path_

    @path.setter
    def path(self, value):
        self.path_ = Path(value) if value is not None else None

    @property
    def shape(self):
        if (self.shape_ is None) and (self.dataset is not None):
            n = self.dataset.RasterXSize
            m = self.dataset.RasterYSize
            return m, n
        else:
            return self.shape_

    @shape.setter
    def shape(self, value):
        self.shape_ = value

    def get_raster(self, band, factor=1):
        # first, check if the band is pertinent
        if band not in self.available_bands:
            print(f'Band {band} not available')
            return None

        # then, check if band is already loaded and with the right shape
        if (band in self.loaded_bands_.keys()) and (self.loaded_bands_[band].shape == self.shape):
            return self.loaded_bands_[band]

        # if the band is a derived band, call the band_math with the formula
        if band in self.calc_bands:
            arr = self.band_math(band, self.calc_bands_[band])
        else:
            # otherwise, try to open the dataset
            ras = self.get_gdal_band(band)
            arr = ras.ReadAsArray(buf_xsize=self.shape[1],
                                  buf_ysize=self.shape[0],
                                  resample_alg=self.resampling)*factor
            # astype('float32')

            # if not successful, it will raise an error
            if arr is None:
                print(f'Band {band} could not be opened')
            else:
                self.loaded_bands_.update({band: arr})

        return arr

    def set_band_math(self, name, fn):
        self.calc_bands_.update({name: fn})

    def band_math(self, name, fn):
        # calc the resulting raster with given formula
        calc_band = fn(self)

        # update the result in the loaded bands dict
        self.loaded_bands_.update({name: calc_band})

        # update the formula
        self.set_band_math(name, fn)

        return calc_band

    def get_gdal_band(self, i):
        if i in self.available_bands:
            return self.dataset.GetRasterBand(i+1)
        else:
            print(f'Band {i} not available')
            return None

    def as_dic(self, bands=None):

        bands = self.available_bands if bands is None else bands
        bands = [bands] if type(bands) is not list else bands

        result = {band: self.get_raster(band) for band in bands if self.get_raster(band) is not None}

        return result

    def as_list(self, bands=None):
        bands = self.available_bands if bands is None else bands

        bands = [bands] if type(bands) is not list else bands
        return [self.get_raster(band) for band in bands if self.get_raster(band) is not None]

    def as_cube(self, bands=None, channels_first=False, squeeze=False):
        lst_bands = self.as_list(bands)
        axis = 0 if channels_first else -1

        if len(lst_bands) > 1:
            cube = np.stack(lst_bands, axis=axis)

        elif len(lst_bands) == 1:
            cube = np.expand_dims(lst_bands[0], axis=axis)

        else:
            return None

        return cube if not squeeze else cube.squeeze()

    def clear(self):
        for band in self.loaded_bands_.keys():
            self.loaded_bands_[band] = None
        self.loaded_bands_ = {}

    def show(self, bands, bright=1., ax=None):
        if ax is None:
            plt.imshow(self.as_cube(bands, squeeze=True)*bright)
        else:
            ax.imshow(self.as_cube(bands, squeeze=True)*bright)

    def save_bands(self, bands, name, no_value=0, dtype=gdal.GDT_Float32):
        fn = self.path.with_name(name).with_suffix('.tif')
        array2raster(str(fn), self.as_cube(bands), self.geo_transform, self.projection, nodatavalue=no_value, dtype=dtype)

    def __del__(self):
        # print(f'Cleaning memory from WNImage')
        self.clear()

    def __getitem__(self, bands):
        lst = self.as_list(bands)
        if len(lst) == 1:
            return lst[0]
        else:
            return lst

    def __repr__(self):
        s = f'WNImage with {len(self.available_bands)} available bands: {self.available_bands} \n'
        s += f'Source: {self.path} \n'
        # s += f'Bands {list(self.loaded_bands_.keys())} loaded in memory'
        return s
        # if self.dataset:
        #     return f'WNImage obj with {len(self.available_bands)} bands'
        # else:
        #     return f'Empty WNImage object'


####################################################################################
class WNSatImage(WNImage):
    dicS2_L1C = {'B1': '_B01.jp2', 'B2': '_B02.jp2', 'B3': '_B03.jp2', 'B4': '_B04.jp2',
                 'B5': '_B05.jp2', 'B6': '_B06.jp2', 'B7': '_B07.jp2', 'B8': '_B08.jp2', 'B8a': '_B8A.jp2',
                 'B9': '_B09.jp2', 'B10': '_B10.jp2', 'B11': '_B11.jp2', 'B12': '_B12.jp2',
                 }

    dicS2_THEIA = {'B2': 'SRE_B2.tif', 'B3': 'SRE_B3.tif', 'B4': 'SRE_B4.tif', 'B5': 'SRE_B5.tif',
                   'B6': 'SRE_B6.tif', 'B7': 'SRE_B7.tif', 'B8': 'SRE_B8.tif', 'B8A': 'SRE_B8A.tif',
                   'B11': 'SRE_B11.tif', 'B12': 'SRE_B12.tif',
                   }

    dicS2_L2A = {'B1': '_B01_60m.jp2', 'B2': '_B02_10m.jp2', 'B3': '_B03_10m.jp2', 'B4': '_B04_10m.jp2',
                 'B5': '_B05_20m.jp2', 'B6': '_B06_20m.jp2', 'B7': '_B07_20m.jp2', 'B8': '_B08_10m.jp2',
                 'B8a': '_B8A_20m.jp2',
                 'B9': '_B09_60m.jp2', 'B11': '_B11_20m.jp2', 'B12': '_B12_20m.jp2',
                 }


    def __init__(self, path, img_dic=None, verbose=True, shape=None):
        super().__init__(None, shape)

        if img_dic is None:
            img_dic = self.dicS2_THEIA

        self.path, self.img_dic, self.verbose = path, img_dic, verbose

        self.datasets = self.open_img() if path is not None else {}

        # initialize with known indices
        self.set_band_math('ndwi', lambda x: self.normalized_difference('B3', 'B8'))
        self.set_band_math('mndwi', lambda x: self.normalized_difference('B3', 'B11'))

    @property
    def data_source(self):
        if len(self.datasets) == 0:
            print(f'No datasets in WNSatImage')
            return None
        else:
            return list(self.datasets.values())[1]

    @property
    def projection(self):
        return self.data_source.GetProjection()

    @property
    def geo_transform(self):
        return self.data_source.GetGeoTransform()

    @property
    def shape(self):
        if (self.shape_ is None) and (len(self.datasets) > 0):
            n = self.datasets[list(self.datasets.keys())[0]].RasterXSize
            m = self.datasets[list(self.datasets.keys())[0]].RasterYSize
            return m, n
        else:
            return self.shape_

    @shape.setter
    def shape(self, value):
        self.shape_ = value

    @property
    def available_bands(self):
        return list(self.datasets.keys()) + self.calc_bands

    def reset_shape(self):
        self.shape_ = None

    def open_gdal_ds(self, name, recursive=True):
        file = search_file([self.path], name, recursive)

        if file is None:
            print(f'File {name} not found in {self.path} and subdirectories')
            return None
        else:
            ds = gdal.Open(str(file))
            if ds is None:
                print(f'Could not open file {file}')

            return ds

    def open_img(self):
        bands = self.img_dic.keys()
        result = {}
        for band_key in bands:
            result.update({band_key: self.open_gdal_ds(self.img_dic[band_key], recursive=True)})
        return result

    def get_gdal_band(self, key):
        if key in self.available_bands:
            return self.datasets[key]
        else:
            print(f'Band {key} not available')
            return None

    def get_raster(self, band, factor=None):
        factor = 1/10000 if factor is None else factor
        return super().get_raster(band, factor=factor)
        # return arr/10000 if arr is not None else None

    def __repr__(self):
        s = f'WNSatImage with {self.available_bands} available bands \n'
        s += f'Source: {self.path} \n'
        s += f'Bands {list(self.loaded_bands_.keys())} loaded in memory'
        return s


####################################################################################
class WNPatchProcessor:
    def __init__(self, img=None, patches_path=None, from_patches=None):

        self.patches_, self.path_patches_, self.format_ = [], [], {}

        self.img = None

        if img is not None:
            self.img = img
        elif patches_path is not None:
            self.load_patches(Path(patches_path))
        elif from_patches is not None:
            self.get_patches(from_patches)

        #todo: assembled img deve ser do tipo WNImage, para poder salvar e fazer o resto

    @classmethod
    def create_from_patches(cls, patches, size, shift, patches_per_row=None, channels_first=None, projection=None,
                            geo_transform=None):
        # If channels_first is None, it signifies that there is only one channel and it is squeezed (must be created)

        proc = cls(from_patches=patches)
        proc.set_format(0, size, shift, channels_first, ppr=patches_per_row, projection=projection,
                        geo_transform=geo_transform)

        return proc

    @property
    def format(self):
        return self.format_

    @property
    def channels_first(self):
        return self.format['channels_first']

    @channels_first.setter
    def channels_first(self, value):
        self.format_.update({'channels_first': value})

    @property
    def bands_string(self):
        s = ''
        for item in self.format['bands']:
            s += str(item)
        return s

    @property
    def patch_height(self):
        if self[0].ndim == 3:
            if self.channels_first:
                return self[0].shape[1]
            else:
                return self[0].shape[0]
        else:
            return self[0].shape[0]

    @property
    def patch_width(self):
        if self[0].ndim == 3:
            if self.channels_first:
                return self[0].shape[2]
            else:
                return self[0].shape[1]
        else:
            return self[0].shape[1]

    @property
    def num_channels(self):
        if self[0].ndim==2:
            return 1
        else:
            if self.channels_first:
                return self[0].shape[0]
            else:
                return self[0].shape[-1]

    @property
    def geo_transform(self):
        if self.img is not None:
            return self.img.geo_transform
        else:
            return self.format['geo_transform']

    @property
    def projection(self):
        if self.img is not None:
            return self.img.projection
        else:
            return self.format['projection']

    def get_patches(self, patches):
        self.clear()

        # if the patches are in form of array, listify the first axis
        if isinstance(patches, np.ndarray):
            self.patches_ = []
            for i in range(patches.shape[0]):
                self.patches_.append(patches[i])
        else:
            self.patches_ = patches

        self.path_patches_ = []

    def set_format(self, bands, size, shift, channels_first, geo_transform=None, projection=None, ppr=None):
        format_ = {
            'bands': bands if type(bands) == list else [bands],
            'size': size,
            'shift': shift,
            'channels_first': channels_first,
            'projection': projection,
            'geo_transform': geo_transform,
            'patches_per_row': ppr
        }
        self.format_ = format_

    def create_patches(self, bands, size, shift, channels_first=False):

        self.set_format(bands, size, shift, channels_first)

        bands = bands if type(bands) == list else [bands]

        cube = self.img.as_cube(bands, channels_first=False)

        dims = (0, 1, 2) if not channels_first else (2, 0, 1)

        num_patches_hor = math.floor(1 + (cube.shape[1] - size) / shift)
        num_patches_ver = math.floor(1 + (cube.shape[0] - size) / shift)

        squares = [np.transpose(cube[i * shift:i * shift + size, j * shift:j * shift + size, :], dims)
                   for i in range(num_patches_ver)
                   for j in range(num_patches_hor)]

        self.patches_ = list(map(np.squeeze, squares))

        # return self.patches_

    def get_visual_patch(self, idx, bright=1., chnls=[3, 2, 1]):
        patch = self[idx]
        if patch is not None:
            if patch.ndim == 2:
                visual_patch = patch
            else:
                dims = (0, 1, 2) if not self.channels_first else (1, 2, 0)
                patch = np.transpose(patch, dims)

                if patch.shape[2] >= 3:
                    visual_patch = patch[:, :, chnls]
                else:
                    visual_patch = patch[:, :, 0]
        else:
            visual_patch = None

        if patch.dtype != 'int32':
            visual_patch = np.where(visual_patch < 0, 0, visual_patch) * bright
            visual_patch = np.where(visual_patch > 1, 1, visual_patch)

        return visual_patch

    def show_item(self, idx, bright=1., ax=None):
        patch = self.get_visual_patch(idx, bright)
        if patch is not None:
            if ax is None:
                plt.imshow(patch)
            else:
                ax.imshow(patch)

    def show_patches(self, first=None, last=None, bright=1., chnls=[3, 2, 1]):
        if len(self) <= 0:
            print(f'No patches to show')
            return

        # adjust first and last patches to the list size
        first = 0 if first is None else (first if first < (len(self)-1) else len(self)-2)
        last = len(self) if last is None else (last if last < len(self) else len(self)-1)
        qty = last-first+1

        # calc the size of the plots
        side = math.ceil(math.sqrt(float(qty)))
        fig, ax = plt.subplots(side, side, figsize=(20, 20))

        ax = ax.reshape(-1)
        for p in range(qty):
            ax[p].imshow(self.get_visual_patch(p+first, bright, chnls=chnls))

    def save_patches(self, path, base_name, ext='npy', fill_nan=None):
        if len(self) == 0:
            print(f'No patches to save')
            return

        path.mkdir(parents=True, exist_ok=True)

        for i, patch in enumerate(self.patches_):

            if fill_nan is not None:
                patch = np.nan_to_num(patch, nan=fill_nan)

            fn = (path / f'{base_name}_{self.bands_string}_{i}').with_suffix('.'+ext)
            # patch = np.where(patch > 1, 1, np.where(patch < 0, 0, patch))

            if ext == 'npy':
                np.save(str(fn), patch, allow_pickle=False)

            elif ext == 'jpg':
                plt.imsave(fn.with_suffix('.png'), patch)

            elif ext == 'png':
                pil_img = PilImg.fromarray(patch)
                pil_img.save(fn, 'PNG')
            
            elif ext == 'torch':
                torch.save(patch, fn)

    def load_patches(self, path, bands=[], size=0, shift=0, base_name='', channels_first=True, in_memory=False):
        self.set_format(bands, size, shift, channels_first)

        imgs_names = [(int(str(file).split('_')[-1].split('.')[0]), str(file)) for file in path.iterdir()
                      if (not file.is_dir()) and (base_name in file.stem)]
        imgs_names.sort()

        # create the list with the files in disk
        self.path_patches_ = [file[1] for file in imgs_names]

        if in_memory:
            self.patches_ = [np.load(file[1]) for file in imgs_names]

        return None

    def assembly_patches(self, channels_first=None):
        if channels_first is not None:
            self.channels_first = channels_first

        if len(self) == 0:
            print(f'No patches to assembly')
            return

        if self.format['patches_per_row'] is None:
            patches_by_row = int(math.sqrt(len(self)))
        else:
            patches_by_row = self.format['patches_per_row']

        patches_by_column = int(len(self)/patches_by_row)

        size_y = self.patch_height * patches_by_column
        size_x = self.patch_width * patches_by_row

        # create the scene array
        if self.channels_first:
            scene = np.ones((self.num_channels, size_y, size_x))
        else:
            scene = np.ones((size_y, size_x, self.num_channels))

        print(f'Creating image shape {scene.shape}')

        patch_size = self.patch_width
        row = 0
        col = 0
        idx = 0

        for patch in self:
            if patch.ndim == 2:
                if self.channels_first:
                    patch = patch[np.newaxis, :, :]
                else:
                    patch = patch[:, :, np.newaxis]

            if idx == patches_by_row:
                row += self.format['shift']
                col = 0
                idx = 0

            if self.channels_first:
                scene[:, row:row + patch_size, col:col + patch_size] = patch
            else:
                scene[row:row + patch_size, col:col + patch_size, :] = patch

            col += self.format['shift']
            idx += 1

        return scene.squeeze()

    def save_scene(self, path, dtype=gdal.GDT_Float32):
        scene = self.assembly_patches()

        array2raster(str(path), scene, self.geo_transform, self.projection, nodatavalue=0, dtype=dtype)

        return scene

    def get_patch_path(self, item):
        if item < len(self.path_patches_):
            return self.path_patches_[item]
        else:
            print(f'Path for item {item} not found')
            return None

    def clear(self):
        if self.img is not None:
            self.img.clear()

        for patch in self.patches_:
            patch = None

        self.patches_ = []

    def patch_as_pil(self, idx):
        patch = self[idx]
        if self.channels_first:
            # send the channels to last dimension
            patch = np.transpose(patch, (1, 2, 0))

        # use just 3 channels
        pi = PilImg.fromarray((patch[:, :, :3] * 255).astype(np.uint8), mode='RGB')

        return pi

    def __len__(self):
        length = len(self.patches_) if len(self.patches_) > 0 else len(self.path_patches_)
        return length

    def __getitem__(self, item):
        # First check if the item is in the range
        if item < len(self):
            # Check if the patches are in memory
            if item < len(self.patches_):
                return self.patches_[item]
            # otherwise, load from disk
            else:
                item_path = Path(self.path_patches_[item])
                if item_path.suffix == '.npy':
                    return np.load(self.path_patches_[item])
                elif item_path.suffix == '.png':
                    return plt.imread(self.path_patches_[item])
        else:
            print(f'Patch {item} not found')
            return None

    def __repr__(self):
        s = f'Patch Processors with {len(self)} patches and {len(self.patches_)} in memory patches \n'
        s += f'Source image:\n{str(self.img) if self.img is not None else "None"}'
        return  s

    def __iter__(self):
        return iter(self.patches_)

    def __del__(self):
        self.clear()


####################################################################################
class WNDataset(torch.utils.data.Dataset):
    def __init__(self, imgs=None, lbls=None, cuda=True, path=None):
        super().__init__()

        self.imgs, self.lbls = None, None
        self.path_ = path

        if path is None:
            self.set_attr('imgs', imgs)
            self.set_attr('lbls', lbls)
        else:
            self.imgs = WNPatchProcessor(patches_path=path/'Images')
            self.lbls = WNPatchProcessor(patches_path=path/'Labels')

        self.cuda = cuda

        self.train_dl, self.valid_dl = None, None

    @property
    def path(self):
        return self.path_

    @property
    def has_labels(self):
        return self.lbls is not None

    def show_item(self, idx, bright=1., ax=None, size=4):
        columns = 2 if self.has_labels else 1
        if ax is None:
            fig, ax = plt.subplots(1, columns, figsize=(size*columns, size))

        if type(ax) == np.ndarray:
            ax = ax.reshape(-1)
            ax[0].set_title('Image')
            self.imgs.show_item(idx, bright=bright, ax=ax[0])
            ax[1].set_title('Label')
            self.lbls.show_item(idx, bright=bright, ax=ax[1])
        else:
            ax.set_title('Image')
            self.imgs.show_item(idx, bright=bright, ax=ax)

    def show_items(self, idxs, bright=1., size=4):
        for idx in idxs:
            self.show_item(idx, bright, size=size)

    def set_attr(self, name, value):
        if value is not None:
            if isinstance(value, WNPatchProcessor) or True:
                if len(value) == 0:
                    print(f'Warning: Creating WNDataset with 0 elements in {name}')
                setattr(self, name, value)
            else:
                print(f'Error: Object {name} is not a WNPatchProcessor: {type(value)} {__name__}')
                setattr(self, name, None)
        else:
            setattr(self, name, None)

    # def set_data(self, imgs, lbls=None):
    #     self.data.set_data(imgs, lbls)

    def create_data_loaders(self, bs, shuffle=True, valid_size=0):
        train_ds, valid_ds = torch.utils.data.random_split(self, (len(self)-valid_size, valid_size))
        self.train_dl = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=shuffle)
        self.valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=bs, shuffle=shuffle)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        x = (self.imgs[item] + 1) / 2
        y = (self.lbls[item] == 1).astype(int) if self.has_labels else 0
        # y = (self.lbls[item] + 1) / 2 if self.has_labels else 0

        if self.cuda:
            return torch.tensor(x, dtype=torch.float32).cuda(), torch.tensor(y, dtype=torch.int64).cuda()
        else:
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)

    def __repr__(self):
        s = f'WNDataset with {len(self)} items. Labels={self.has_labels}'
        return s


####################################################################################
class WNLearner:
    def __init__(self, dataset, model):
        self.dataset, self.model = dataset, model
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.losses, self.accuracies = ([], []), ([], [])
        self.checkpoints = []

    def train(self, lr=0.0001, epochs=1, new_model=None, show_each=10):

        self.model = self.model if new_model is None else new_model
        self.model.cuda()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Start the training loop
        start = time.time()

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)

            for phase_value, phase in enumerate(['train', 'valid']):
                if phase == 'train':
                    self.model.train(True)  # Set training mode = true
                    data_loader = self.dataset.train_dl
                else:
                    self.model.train(False)  # Set model to evaluate mode
                    data_loader = self.dataset.valid_dl

                # init variables
                running_loss = 0.0
                running_acc = 0.0
                step = 0

                # iterate over data
                for step, (x, y) in enumerate(data_loader):

                    if phase == 'train':
                        # zero the gradients
                        opt.zero_grad()
                        outputs = self.model(x)
                        loss = self.loss_fn(outputs, y)

                        # the backward pass frees the graph memory, so there is no
                        # need for torch.no_grad in this training pass
                        loss.backward()
                        opt.step()
                        # scheduler.step()
                    else:
                        with torch.no_grad():
                            outputs = self.model(x)
                            loss = self.loss_fn(outputs, y.long())

                    # stats - whatever is the phase
                    acc = self.accuracy(outputs, y)

                    running_acc  += acc*data_loader.batch_size
                    running_loss += loss*data_loader.batch_size

                    if step % show_each == 0:
                        print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                        # print(torch.cuda.memory_summary())

                epoch_loss = running_loss / len(data_loader.dataset)
                epoch_acc = running_acc / len(data_loader.dataset)

                # print('Epoch {}/{}'.format(epoch, epochs - 1))
                print('-' * 10)
                print('{} Loss: {:.4f} Acc: {}  Time{:.0f}m {:.0f}s'
                      .format(phase, epoch_loss, epoch_acc, (time.time() - start) // 60, (time.time() - start) % 60))
                print('-' * 10)

                self.losses[phase_value].append(epoch_loss)
                self.accuracies[phase_value].append(epoch_acc)

        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    @staticmethod
    def accuracy(pred_b, y_b):
        return (pred_b.argmax(dim=1) == y_b.cuda()).float().mean()

    @property
    def models_path(self):
        model_path = self.dataset.path/'models'
        model_path.mkdir(exist_ok=True) if not model_path.exists() else None

        return model_path

    def predict_item(self, idx, dataset=None):
        x, _ = self.dataset[idx] if dataset is None else dataset[idx]

        with torch.no_grad():
            probs = self.model(x.unsqueeze(0)).squeeze().cpu()
        return torch.argmax(probs, axis=0).int(), probs

    def show_prediction(self, idx, bright=1.):
        # display input and original label
        fig, ax = plt.subplots(1, 5, figsize=(15, 5))

        self.dataset.show_item(idx, bright, ax=ax[0:2])

        # display predictions
        pred = self.predict_item(idx)
        ax[2].set_title('Prediction')
        ax[2].imshow(pred[0].numpy())
        ax[3].set_title('Prob1')
        ax[3].imshow(pred[1][0].numpy())
        ax[4].set_title('Prob2')
        ax[4].imshow(pred[1][1].numpy())

    def show_predictions(self, idxs, bright=1.):
        for idx in idxs:
            self.show_prediction(idx, bright)

    def plot_losses(self):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(self.losses[0], label='Train loss')
        ax[0].plot(self.losses[1], label='Valid loss')
        ax[0].legend()
        ax[1].plot(self.accuracies[0], label='Train Acc')
        ax[1].plot(self.accuracies[1], label='Valid Acc')
        ax[1].legend()

    def save_checkpoint(self, name):
        checkpoint_name = (self.models_path/name).with_suffix('.pth')
        self.checkpoints.append(checkpoint_name)
        torch.save(self.model.state_dict(), checkpoint_name)

    def load_checkpoint(self, checkpoint):
        path = self.checkpoints[checkpoint] if type(checkpoint) == int else \
            (self.models_path/checkpoint).with_suffix('.pth')
        self.checkpoints.append(path)
        print(f'Loading weights at {path}')
        self.model.load_state_dict(torch.load(path))

    def predict_data(self, dataset):
        preds = []
        for idx in range(len(dataset)):
            preds.append(self.predict_item(idx, dataset)[0])

        return preds

    def __repr__(self):
        s = f'Learner with {len(self.checkpoints)} checkpoint.\n'
        s += f'Last checkpoint at {self.checkpoints[-1] if len(self.checkpoints) > 0 else "None"}\n'
        s += f'Object at: {super().__repr__()}'
        return s



