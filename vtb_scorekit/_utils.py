# -*- coding: utf-8 -*-

import pandas as pd
import io
import os


def make_header(s, l):
    return(f"{f' {s} ':-^{l}}")


def fig_to_excel(fig, ws, row=0, col=0, scale=1):
    if fig is not None:
        imgdata = io.BytesIO()
        fig.savefig(imgdata, format='png', bbox_inches="tight")
        ws.insert_image(row, col, '', {'image_data': imgdata, 'x_scale': scale, 'y_scale': scale})


def adjust_cell_width(ws, df):
    descr_len = sum([1 for f in df.columns if not pd.api.types.is_numeric_dtype(df[f])])
    ws.set_column(0, 0 + descr_len, 30)
    ws.set_column(1 + descr_len, df.shape[1], 15)


def add_suffix(f):
    if f.endswith('_WOE'):
        return f
    return f + '_WOE'


def rem_suffix(f):
    if f.endswith('_WOE'):
        return f[:-4]
    return f


def is_cross_name(f):
    return f.startswith('cross_') and '&' in f


def cross_name(f1, f2, add_woe=False):
    name = f'cross_{f1}&{f2}'
    return add_suffix(name) if add_woe else name


def cross_split(f):
    f = rem_suffix(f)
    return f[6:].split('&') if is_cross_name(f) else [f, '']


def add_ds_folder(ds, file_name):
    if ds is None or not os.path.exists(ds.result_folder + os.path.dirname(file_name)):
        return file_name
    return ds.result_folder + file_name

