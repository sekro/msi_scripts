#!/usr/bin/env python3

"""
Script to detect cells in brightfield images and to extract data from co-registered MS images
Sebastian Krossa 2021-2022
NTNU Trondheim
sebastian.krossa@ntnu.no
"""

import cv2
import gc
import os
import numpy as np
import pandas as pd
import matplotlib
import time
from matplotlib import pyplot as plt
from collections import namedtuple
from scipy.optimize import curve_fit

matplotlib.use('pdf')

plot_profiles = True

inflection_type = namedtuple('inflection_type',
                             ['x', 'y', 'slope', 'b'],
                             defaults=[None, None, None, None])
fit_type = namedtuple('fit_type',
                      ['a', 'b', 'c', 'i1', 'i2'],
                      defaults=[None, None, None, None, None])
cell_type = namedtuple('cell_type',
                       ['id', 'single_cell', 'area', 'img', 'profile_x', 'profile_y', 'fit_x', 'fit_y', 'profile_x_xvals', 'profile_y_xvals', 'scale'],
                       defaults=[None, None, None, None, None, None, None, None, None, None, None])

# set this to the path containing the images & scale file
# script output will be saved into subfolders in this folder
base_path_data = ''


def add_fit_plots(ax, x, d, scale_d1):
    """
    add plots of the fitted gauss & derivates & slope at inflection points to axis
    :param ax: matplotlib.Axes
    :param x: x values
    :param d: gauss parameter - namedtuple fit_type
    :param scale_d1: float - scale factor of derivative
    :return: None
    """
    ax.plot(x, gauss(x, d.a, d.b, d.c), label='f(x) - gauss fit', color='#2271b2')
    ax.plot(x, np.abs(gauss_dx(x, d.a, d.b, d.c)) * scale_d1, color='#f748a5',
            label="|f'(x)|, scaled by factor {} for better visuability".format(scale_d1))
    ax.plot(x, linear(x, d.i1.slope, d.i1.b), label='slope at infl p1', linestyle='dashed', color='#359b73')
    ax.plot(x, linear(x, d.i2.slope, d.i2.b), label='slope at infl p2', linestyle='dashed', color='red')
    ax.axvline(x=d.i1.x, color='grey', linestyle='dotted')
    ax.axvline(x=d.i2.x, color='grey', linestyle='dotted')


def get_fit_data(profile, x):
    """
    call fitting function and collect data for fit_type namedtuple to return gauss parameter and inflection points
    :param profile: the "to-be-fitted" data
    :param x: respective x-values
    :return: namedtuple fit_type
    """
    popt, pcov = do_fit(profile, x)
    if popt is not None:
        a, b, c = popt
        inflp = []
        inflp.append(b + c)
        inflp.append(b - c)
        _tinfl = []
        for ip in inflp:
            _y = gauss(ip, a, b, c)
            _slope = gauss_dx(ip, a, b, c)
            _b = get_b(ip, _y, _slope)
            _tinfl.append(inflection_type(x=ip, y=_y, slope=_slope, b=_b))
        return fit_type(a=a, b=b, c=c, i1=_tinfl[0], i2=_tinfl[1])
    else:
        return fit_type(i1=inflection_type(), i2=inflection_type())


def do_fit(y, x):
    """
    performs start parameter estimation and gauss fitting of y
    :param y: to be fitted y data
    :param x: x values
    :return: return values from scipy - curve_fit
    """
    n = len(y)
    if n > 0:
        # estimation of start values
        a_start = y.max()
        b_start = x[y.argmax()] / 4
        c = sum(y * (x - b_start) ** 2) / (1000 * n)
        try:
            ret_val = curve_fit(gauss, x, y, p0=[a_start, b_start, c])
            return ret_val
        # lazy - broad exception...
        except:
            return None, None
    else:
        return None, None


# the functions used for fitting & plotting
def gauss(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


def gauss_dx(x, a, b, c):
    return -a * (x - b) * np.exp(-(x - b) ** 2 / (2 * c ** 2)) / (c ** 2)


def linear(x, m, b):
    return m * x + b


def get_b(x, y, m):
    return y - m * x


def save_cells_as_pandas(cells, path):
    """
    Does some formating and sorting into pandas.DataFrame of all cell data and saves it to disk as excel file and image
    file into subfolders under path
    :param cells: list of namedtuple cell_type
    :param path: output path
    :return: None
    """
    single_cell_path = os.path.join(path, 'cell_imgs/single')
    cell_path = os.path.join(path, 'cell_imgs/not_single')
    if not os.path.exists(single_cell_path):
        os.makedirs(single_cell_path)
    if not os.path.exists(cell_path):
        os.makedirs(cell_path)
    _td = {
        'ID': [],
        'Single cell': [],
        'cell bounding box area [µm^2]': [],
        'img': [],
        'profile_x_y_values [%]': [],
        'profile_y_y_values [%]': [],
        'profile_x_x_values [µm]': [],
        'profile_y_x_values [µm]': [],
        'fit_x_a': [],
        'fit_x_b': [],
        'fit_x_c': [],
        'fit_y_a': [],
        'fit_y_b': [],
        'fit_y_c': [],
        'x_inflection_p1 [µm]': [],
        'x_inflection_p2 [µm]': [],
        'y_inflection_p1 [%]': [],
        'y_inflection_p2 [%]': [],
        'slope_x1 [%/µm]': [],
        'slope_x2 [%/µm]': [],
        'slope_y1 [%/µm]': [],
        'slope_y2 [%/µm]': []
    }
    for c in cells:
        if c.img.size > 0:
            _td['ID'].append(c.id)
            _td['Single cell'].append(c.single_cell)
            _td['cell bounding box area [µm^2]'].append(c.area)
            _td['img'].append('ID_{}.png'.format(c.id))
            _td['profile_x_y_values [%]'].append(c.profile_x)
            _td['profile_y_y_values [%]'].append(c.profile_y)
            _td['profile_x_x_values [µm]'].append(c.profile_x_xvals)
            _td['profile_y_x_values [µm]'].append(c.profile_y_xvals)
            _td['fit_x_a'].append(c.fit_x.a)
            _td['fit_x_b'].append(c.fit_x.b)
            _td['fit_x_c'].append(c.fit_x.c)
            _td['fit_y_a'].append(c.fit_y.a)
            _td['fit_y_b'].append(c.fit_y.b)
            _td['fit_y_c'].append(c.fit_y.c)
            _td['x_inflection_p1 [µm]'].append(c.fit_x.i1.x)
            _td['x_inflection_p2 [µm]'].append(c.fit_x.i2.x)
            _td['y_inflection_p1 [%]'].append(c.fit_y.i1.x)
            _td['y_inflection_p2 [%]'].append(c.fit_y.i2.x)
            _td['slope_x1 [%/µm]'].append(c.fit_x.i1.slope)
            _td['slope_x2 [%/µm]'].append(c.fit_x.i2.slope)
            _td['slope_y1 [%/µm]'].append(c.fit_y.i1.slope)
            _td['slope_y2 [%/µm]'].append(c.fit_y.i2.slope)
            if c.single_cell:
                cell_p = os.path.join(single_cell_path, 'ID_{}.png'.format(c.id))
            else:
                cell_p = os.path.join(cell_path, 'ID_{}.png'.format(c.id))
            cv2.imwrite(cell_p, c.img)
    _df = pd.DataFrame(_td)
    _df.to_excel(os.path.join(path, 'out.xlsx'))


def get_file_dict(bp):
    """
    Finds and sorts files inside bp for processing
    :param bp: basepath with data / image files
    :return: dictionary with files
    """
    _ret_dict = {}
    for _p, _sds, _fs in os.walk(bp):
        if _p == bp:
            for _f in _fs:
                if '_type_' in _f:
                    _fn = _f.split('_type_')[0]
                    _ft = _f.split('_type_')[1].split('.')[0]
                    if _fn not in _ret_dict:
                        _ret_dict[_fn] = {}
                    _ret_dict[_fn][_ft] = os.path.join(_p, _f)
    return _ret_dict


def get_crop_box(bfr_f, off=None):
    """
    Performs color thresholding and contour detection to find "red box" in image for croping the data part
    :param bfr_f: string - path to brightfield image with "red box"
    :param off: integer - offset in px to "shrink" crop box to avoid problematic images borders
    :return: tuple(y1, y2, x1, x2, contour)
    """
    img = cv2.imread(bfr_f, cv2.IMREAD_COLOR)
    if 'T30' in bfr_f or 'T39' in bfr_f or 'E29' in bfr_f:
        print('special')
        th1 = cv2.inRange(img, (0, 0, 130), (85, 85, 255))
    else:
        th1 = cv2.inRange(img, (0, 0, 250), (0, 0, 255))
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 94, 213), 2)
    last_w = np.inf
    last_h = np.inf
    last_x = 0
    last_y = 0
    last_a = 0
    ret_c = None
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        #print(x, y, h, w)
        if a > last_a:
            last_w = w
            last_h = h
            last_x = x
            last_y = y
            ret_c = c
            last_a = a
    if off is None:
        off = round(last_w * 0.05)
    return last_y + off, last_y + last_h - off, last_x + off, last_x + last_w - off, ret_c


def get_scales(scalep, i_dict, fnc='Filnavn:', sc='Skala (px/µm):'):
    """
    Extract image scale values from excel file
    :param scalep: string path to excel file
    :param i_dict: dictionary as returned by get_file_dict()
    :param fnc: string - name of the file name col in excel file
    :param sc: string - name of the scale value col in excel file
    :return: dictionary as returned by get_file_dict() with added 'scale' values
    """
    _df = pd.read_excel(scalep)
    for _ri, _rd in _df.iterrows():
        if _rd[fnc] in i_dict:
            i_dict[_rd[fnc]]['scale'] = _rd[sc]
        else:
            print('{} not in i_dict'.format(_rd[fnc]))
    return i_dict


def get_mask(img, use_hist_equal, use_otsu, use_adaptive_thr, bin_thr_cut):
    _gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if use_hist_equal:
        _gray = cv2.equalizeHist(_gray)
    if use_otsu:
        _blur = cv2.GaussianBlur(_gray, (5, 5), 0)
        ret, _th1 = cv2.threshold(_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _blur = cv2.GaussianBlur(_gray, (5, 5), 0)
        if use_adaptive_thr:
            _th1 = cv2.adaptiveThreshold(_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 2)
        else:
            ret, _th1 = cv2.threshold(_blur, bin_thr_cut, 255, cv2.THRESH_BINARY)
    return ret, _th1


def draw_cell_info_on_imgs(cell_id, img, img_msi, bbox, bbox_off_coord):
    x, y, w, h = bbox
    _c_x1, _c_x2, _c_hx, _c_y1, _c_y2, _c_hy = bbox_off_coord
    cv2.rectangle(img, (_c_x1, _c_y1), (_c_x2, _c_y2), (178, 113, 34), 2)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 94, 213), 2)
    cv2.rectangle(img_msi, (_c_x1, _c_y1), (_c_x2, _c_y2), (178, 113, 34), 2)
    cv2.rectangle(img_msi, (x, y), (x + w, y + h), (0, 94, 213), 2)
    cv2.putText(img, 'ID{}'.format(cell_id), (_c_x1 + 2, _c_hy), cv2.FONT_HERSHEY_DUPLEX, 0.6, (178, 113, 34), 1,
                cv2.LINE_AA)
    cv2.putText(img_msi, 'ID{}'.format(cell_id), (_c_x1 + 2, _c_hy), cv2.FONT_HERSHEY_DUPLEX, 0.6, (178, 113, 34), 1,
                cv2.LINE_AA)


def extract_profile(cropped_msi, scale):
    # this is on purpose - cropped_msi is np.uint8 - don't multiply first!
    prof = (cropped_msi / 255) * 100
    prof_xvals = np.array([i / scale for i in range(len(prof))])
    return prof, prof_xvals


def generate_imgs_and_save_output(cells, img, img_clean, img_msi, img_msi_clean, th1, basepath, cell_detect_img, msi_img,
                                  name, scale, scale_bar_um, plot_prof, prof_same_x):
    if not os.path.exists(os.path.join(basepath, 'out/{}'.format(name))):
        os.makedirs(os.path.join(basepath, 'out/{}'.format(name)))
    _bar_start_x = 50
    _bar_y = 20
    _bar_stop_x = _bar_start_x + int(round(scale * scale_bar_um))
    _bar_txt_sz = cv2.getTextSize('{} µm'.format(scale_bar_um), cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
    _bar_txt_x = int(round((_bar_start_x + _bar_stop_x - _bar_txt_sz[0]) / 2))
    _bar_txt_y = _bar_y + _bar_txt_sz[1] + 2
    cv2.line(img_clean, (_bar_start_x, _bar_y), (_bar_stop_x, _bar_y), (0, 0, 255), 1, cv2.LINE_AA)
    cv2.line(img_msi_clean, (_bar_start_x, _bar_y), (_bar_stop_x, _bar_y), (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img_clean, '{} um'.format(scale_bar_um), (_bar_txt_x, _bar_txt_y), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img_msi_clean, '{} um'.format(scale_bar_um), (_bar_txt_x, _bar_txt_y), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(basepath, 'out/{}/{}_type_{}_threshold_img.png'.format(name, name, cell_detect_img)), th1)
    cv2.imwrite(os.path.join(basepath, 'out/{}/{}_type_{}_overview.png'.format(name, name, cell_detect_img)), img)
    cv2.imwrite(os.path.join(basepath, 'out/{}/{}_type_{}_overview_no_overlay.png'.format(name, name, cell_detect_img)),
                img_clean)
    cv2.imwrite(os.path.join(basepath, 'out/{}/{}_type_{}_overview.png'.format(name, name, msi_img)), img_msi)
    cv2.imwrite(os.path.join(basepath, 'out/{}/{}_type_{}_overview_no_overlay.png'.format(name, name, msi_img)),
                img_msi_clean)
    n = len(cells)
    print('Found {} cells after filtering in {}'.format(n, name))
    save_cells_as_pandas(cells, os.path.join(basepath, 'out/{}'.format(name)))
    if plot_prof:
        plot_profiles(cells=cells, path=os.path.join(basepath, 'out/{}'.format(name)), same_x=prof_same_x)


def get_cells_from_image(name, ide, basepath, plot_prof=True, prof_same_x=False, use_otsu=True, use_hist_equal=True,
                         use_adaptive_thr=True,
                         scale_bar_um=200, bin_thr_cut=127,
                         cell_detect_img='BFnr', msi_img='MSnr', offset_um=20, single_cell_max_box_um=50,
                         max_cell_box_area_um2=50000, filter_shape_low=0.25, filter_shape_high=4):
    """
    Runs thresholding & cell detection, cell image generation, calls fit function, save & plot functions
    :param name: string - name of this image set
    :param ide: dict with meta data
    :param basepath: string - basepath for output and data
    :param plot_prof: bool - plot a figure of profiles for each cell
    :param prof_same_x: bool - use same x axes for all profiles
    :param use_otsu: bool use Otsu thresholding
    :param use_hist_equal: bool - apply histogram equalization prio thresholding
    :param use_adaptive_thr: bool - use adaptive thresholding (only effective if use_otsu = False)
    :param scale_bar_um: int value of µm of scale bar used
    :param bin_thr_cut: int value for fix manual threshold value
    :param cell_detect_img: string - dictionary key for cell detection image
    :param msi_img: string - dictionary key for MS image
    :param offset_um: int - offset around cell bounding box
    :param single_cell_max_box_um: int - single_cell_max_box_um ** 2 gets used to filter single cells
    :param max_cell_box_area_um2: int - coarse filter of detected contours
    :param filter_shape_low: float - coarse filter - lower aspect ratio bound of cell bounding box
    :param filter_shape_high: float - coarse filter - upper aspect ratio bound of cell bounding box
    :return: tuple(cells, cell_area_fraction, cell_area_fraction_large_box, cell_area_by_thr)
    """
    px_off = int(round(ide['scale'] * offset_um))
    max_cell_a_px = int(round(ide['scale'] * max_cell_box_area_um2))
    single_cell_max_box_px = int(round((ide['scale'] * single_cell_max_box_um) ** 2))
    _y1, _y2, _x1, _x2 = ide['crop']
    _img = cv2.imread(ide[cell_detect_img], cv2.IMREAD_COLOR)[_y1:_y2, _x1:_x2]
    _img_msi = cv2.imread(ide[msi_img], cv2.IMREAD_COLOR)[_y1:_y2, _x1:_x2]
    _img_clean = _img.copy()
    _img_msi_clean = _img_msi.copy()
    _gray_msi = cv2.cvtColor(_img_msi, cv2.COLOR_BGR2GRAY)
    # get the threshold img aka mask
    _ret, _th1 = get_mask(_img, use_hist_equal, use_otsu, use_adaptive_thr, bin_thr_cut)
    # calc cell area / confluence via mask
    cell_area_by_thr = np.count_nonzero(_th1 == 0) / (_th1.shape[0] * _th1.shape[1])

    # run contour detection
    _contours, _hierarchy = cv2.findContours(_th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw all contours onto imgs
    cv2.drawContours(_img, _contours, -1, (0, 94, 213), 2)
    cv2.drawContours(_img_msi, _contours, -1, (0, 94, 213), 2)

    # start with "cell detection" from contours
    cells = []
    total_cell_area = 0
    total_cell_area_large_box = 0
    for i, _c in enumerate(_contours):
        x, y, w, h = cv2.boundingRect(_c)
        a = w * h
        a_large_box = (w + 2 * px_off) * (h + 2 * px_off)
        a_um = (w / ide['scale']) * (h / ide['scale'])
        # "coarse" filter of contour by aspect ratio and size of bounding box
        if (filter_shape_low < (w / h) < filter_shape_high) and h > 0 and w > 0 and a <= max_cell_a_px:
            total_cell_area += a
            total_cell_area_large_box += a_large_box
            # the offset bounding box parameters
            _c_x1 = x - px_off
            _c_x2 = x + w + px_off
            _c_hx = int(round(x + w / 2))
            _c_y1 = y - px_off
            _c_y2 = y + h + px_off
            _c_hy = int(round(y + h / 2))
            # check values are inside ms img
            max_h, max_w = _gray_msi.shape
            if _c_x1 < 0:
                _c_x1 = 0
            if _c_y1 < 0:
                _c_y1 = 0
            if _c_x2 > max_w:
                _c_x2 = max_w
            if _c_y2 > max_h:
                _c_y2 = max_h
            if _c_hy >= max_h:
                _c_hy = max_h - 1
                print('WARNING: Cell ID {} in image {} has y center profile out of bounds'.format(i, name))
            if _c_hx >= max_w:
                _c_hx = max_h - 1
                print('WARNING: Cell ID {} in image {} has x center profile out of bounds'.format(i, name))
            # extract profiles
            prof_x, prof_x_xvals = extract_profile(_gray_msi[_c_y1:_c_y2, _c_hx], ide['scale'])
            prof_y, prof_y_xvals = extract_profile(_gray_msi[_c_hy, _c_x1:_c_x2], ide['scale'])
            if a <= single_cell_max_box_px:
                _single_cell = True
            else:
                _single_cell = False
            cells.append(cell_type(id=i, single_cell=_single_cell, area=a_um, img=_gray_msi[_c_y1:_c_y2, _c_x1:_c_x2],
                                   profile_x=prof_x, profile_y=prof_y, profile_x_xvals=prof_x_xvals,
                                   profile_y_xvals=prof_y_xvals,
                                   fit_x=get_fit_data(prof_x, prof_x_xvals), fit_y=get_fit_data(prof_y, prof_y_xvals),
                                   scale=ide['scale']))
            draw_cell_info_on_imgs(i, _img, _img_msi, (x, y, w, h), (_c_x1, _c_x2, _c_hx, _c_y1, _c_y2, _c_hy))
    generate_imgs_and_save_output(cells, _img, _img_clean, _img_msi, _img_msi_clean, _th1, basepath, cell_detect_img,
                                  msi_img, name, ide['scale'], scale_bar_um, plot_prof, prof_same_x)
    cell_area_fraction = total_cell_area / (_img.shape[0] * _img.shape[1])
    cell_area_fraction_large_box = total_cell_area_large_box / (_img.shape[0] * _img.shape[1])
    return cells, cell_area_fraction, cell_area_fraction_large_box, cell_area_by_thr


def plot_profiles(cells, path, scale_d1=10, same_x=False):
    """
    Generates the plot for each cell in cells and writes it as pdf to disk
    :param cells: list of namedtuple cell_type
    :param path: output path
    :param scale_d1: scaling of the first derivative for better visuals
    :param same_x: use same x axis for all plots
    :return: None
    """
    plot_n = len(cells)
    out_path = os.path.join(path, 'profiles')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    _start = time.time()
    _max_x = 0
    for i in range(plot_n):
        if cells[i].img.size > 0:
            if cells[i].profile_x_xvals.max() > _max_x:
                _max_x = cells[i].profile_x_xvals.max()
            if cells[i].profile_y_xvals.max() > _max_x:
                _max_x = cells[i].profile_y_xvals.max()
    for i in range(plot_n):
        if cells[i].img.size > 0:
            fig, axs2 = plt.subplots(1, 3, figsize=(18, 8))
            axs2[1].set_ylim([-1, 120])
            axs2[2].set_ylim([-1, 120])
            if same_x:
                _px_xmax = _max_x
                _py_xmax = _max_x
            else:
                _px_xmax = cells[i].profile_x_xvals.max()
                _py_xmax = cells[i].profile_y_xvals.max()
            axs2[1].set_xlim([-10, _px_xmax + 10])
            axs2[2].set_xlim([-10, _py_xmax + 10])
            axs2[0].imshow(cells[i].img, cmap='gray')
            axs2[0].set_title('Img of spot ID{}'.format(i))
            # relabel
            h, w = cells[i].img.shape
            y_ticks = np.arange(0, h / cells[i].scale, 10) * cells[i].scale
            y_labels = [str(round(i)) for i in np.arange(0, h / cells[i].scale, 10)]
            x_ticks = np.arange(0, w / cells[i].scale, 10) * cells[i].scale
            x_labels = [str(round(i)) for i in np.arange(0, w / cells[i].scale, 10)]
            axs2[0].set_xticks(x_ticks, labels=x_labels)
            axs2[0].set_yticks(y_ticks, labels=y_labels)
            axs2[0].set_xlabel('Position [µm]')
            axs2[0].set_ylabel('Position [µm]')
            axs2[1].plot(cells[i].profile_x_xvals, cells[i].profile_x, color='tab:grey', label='profile')
            axs2[1].set_title('Center profile in x direction')
            axs2[1].set_xlabel('Postion [µm]')
            axs2[1].set_ylabel('Intensity [%]')
            if cells[i].fit_x.a is not None:
                add_fit_plots(axs2[1], cells[i].profile_x_xvals, cells[i].fit_x, scale_d1)
                axs2[1].legend()

            axs2[2].plot(cells[i].profile_y_xvals, cells[i].profile_y, color='tab:grey', label='profile')
            axs2[2].set_title('Center profile in y direction')
            axs2[2].set_xlabel('Postion [µm]')
            axs2[2].set_ylabel('Intensity [%]')
            if cells[i].fit_y.a is not None:
                add_fit_plots(axs2[2], cells[i].profile_y_xvals, cells[i].fit_y, scale_d1)
                axs2[2].legend()
            fig.savefig(os.path.join(out_path, 'profile_and_fit_ID{}.pdf'.format(i)), format='pdf')
            fig.clear()
            plt.close('all')
            gc.collect()
    _stop = time.time()
    print('matplotlib plot time: {} s for {} plots -> {} fps'.format(_stop-_start, plot_n, plot_n/(_stop-_start)))


if __name__ == "__main__":
    if os.path.exists(base_path_data):
        # get all files & folders for processing
        i_dict = get_file_dict(base_path_data)
        # some setup
        _run_cropping = True
        plt.ioff()
        # loop over all file∕image sets
        for _i, _is in i_dict.items():
            # find the "red box" defining the "data" area
            # off parameter defines "inner" offset from red border for cropping in px
            y1, y2, x1, x2, c = get_crop_box(_is['BFr'], off=25)
            # if c is none no crop box was found
            if c is not None:
                # T31 and T32 needs manual fix cause positions don't match
                if _run_cropping:
                    # here the actual cropping of the images happens
                    for _itype, _ipath in _is.items():
                        _img = cv2.imread(_ipath, cv2.IMREAD_COLOR)
                        _img = _img[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(base_path_data, 'crop/{}_type_{}.png'.format(_i, _itype)), _img)
                _is['crop'] = (y1, y2, x1, x2)
            else:
                print('Something wrong with {}'.format(_i))
        # extract the px to µm scale values
        i_dict = get_scales(os.path.join(base_path_data, 'Skala.xlsx'), i_dict)
        _idx = []
        _cf = []
        _cf_l = []
        _cf_th = []
        # set the bounding box offset for each cell during cell detection
        offset_um = 20
        # exclude E13-E15 and E19-E30 from profile plotting.
        # no_plot_prof_list = []
        no_plot_prof_list = ['E{}'.format(i) for i in range(13, 16)] + ['E{}'.format(i) for i in range(19, 31)]
        for _i, _is in i_dict.items():
            if _i in no_plot_prof_list:
                plot_prof = False
            else:
                plot_prof = True
            # runs the cell detection, data extraction & fitting
            _cells, _cell_frac, _cell_frac_large_box, _cell_th = get_cells_from_image(_i, _is, base_path_data, plot_prof=False,
                                                                            use_otsu=True,
                                                                            use_hist_equal=False,
                                                                            prof_same_x=True,
                                                                            cell_detect_img='BFr',
                                                                            msi_img='MSnr', offset_um=offset_um,
                                                                            filter_shape_low=0, filter_shape_high=1000)
            _idx.append(_i)
            _cf.append(_cell_frac)
            _cf_l.append(_cell_frac_large_box)
            _cf_th.append(_cell_th)
            plt.close('all')
        # get the cell density and save to disk
        _df = pd.DataFrame(data={'% area with cells': _cf, '% area with cells inkl {} um offset'.format(offset_um): _cf_l,
                                 '% area with cells using px': _cf_th},
                           index=_idx)
        _df.to_excel(os.path.join(base_path_data, 'out/area_with_cells.xlsx'))
        #plt.show()
    else:
        print('No data found - did you set base_path_data correctly (currently set to "{}")'.format(base_path_data))