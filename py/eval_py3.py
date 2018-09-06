#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report prediction metrics (py3)."""

from __future__ import (
    division, absolute_import, print_function, unicode_literals)
import os
import json
import argparse
from io import open
from math import radians, cos, asin, sqrt

import six
from six.moves import cPickle, xrange
import numpy as np

def haversine_dist(loc1, loc2):
    """Calculate distance between locations (haversine formula)."""
    def hav(theta):
        return (1 - cos(theta)) / 2.0

    lat1, lon1, lat2, lon2 = map(radians, loc1 + loc2)
    loc_hav = hav(lat2 - lat1)  + cos(lat1) * cos(lat2) * hav(lon2 - lon1)
    return 2 * 6371 * asin(sqrt(loc_hav))


def get_label_info(path):
    ext = os.path.splitext(path)[-1]
    if ext == '.txt':
        label, lbl_type = [], []
        with open(path, 'rt', encoding='utf-8') as fin:
            for line in fin:
                if line[0] == '#':
                    break
                a = line.strip().split('\t', 1)[0]
                label.append(a[1:])
                lbl_type.append(a[0])
    elif ext == '.pkl':
        with open(path, 'rb') as fin:
            obj = cPickle.load(fin)
        label2id = obj[-1]
        id2label = {v: k for k, v in six.iteritems(label2id)}
        label = [id2label[i] for i in obj[2]]
        train_index, val_index, test_index = obj[3:6]
        lbl_type = [''] * len(label)
        for idx in train_index:
            lbl_type[idx] = '+'
        for idx in val_index:
            lbl_type[idx] = '*'
        for idx in test_index:
            lbl_type[idx] = '?'
    else:
        raise NotImplementedError()
    return label, lbl_type


def main():
    parser = argparse.ArgumentParser(
        description='evalulate prediction results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--user_loc', help='user location for acc@161')
    parser.add_argument('train_path', help='training path')
    parser.add_argument('pred_path', help='prediction path')
    parser.add_argument('geo_path', help='geolocation path')
    args = parser.parse_args()

    with open(args.geo_path, 'rt', encoding='utf-8') as fin:
        geoloc = json.load(fin)

    label, lbl_type = get_label_info(args.train_path)
    label_set = set(label)
    n_lbls = len(label_set)
    hit = [0] * n_lbls
    test = 0

    cnt = -1
    none_cnt = 0
    dist_errors = []
    user_pred = []
    with open(args.pred_path, 'rt', encoding='utf-8') as fin:
        line = fin.readline()
        tags = line.strip().split(' ')
        for line in fin:
            cnt += 1
            lbl = label[cnt]
            lbl_t = lbl_type[cnt]
            if lbl_t != '?':
                continue
            a = line.strip().split(' ')
            b = {}
            for i in xrange(n_lbls):
                b[tags[i]] = float(a[i])
            c = sorted(b.items(), key=lambda x_y: x_y[1], reverse=True)
            assert c[0][1] > 0.
            test += 1
            for i in xrange(n_lbls):
                if c[i][0] == lbl and c[i][1] != 0.:
                    hit[i] += 1
            pred_lbl = c[0][0]
            user_pred.append((cnt, pred_lbl))
            if c[0][1] == 0.0:
                pred_lbl = None
                none_cnt += 1
            if pred_lbl == lbl:
                dist_errors.append(0.)
            else:
                if pred_lbl != None:
                    geo_truth = geoloc[lbl]
                    geo_pred = geoloc[pred_lbl]
                    dist_errors.append(haversine_dist(geo_truth, geo_pred))
    print('None:', none_cnt)
    print(test, len(user_pred))

    print('Acc = {:.4f}'.format(hit[0] / test))
    print('Acc@3 = {:.4f}'.format((hit[0] + hit[1] + hit[2]) / test))
    print('Mean = {:.2f}'.format(np.mean(dist_errors)))
    print('Median = {:.2f}'.format(np.median(dist_errors)))

    if args.user_loc:
        with open(args.user_loc, 'rt', encoding='utf-8') as fin:
            user_loc = json.load(fin)
        grid_dists = []
        for user, pred in user_pred:
            geo_truth = user_loc[str(user)]
            geo_pred = geoloc[pred]
            grid_dists.append(haversine_dist(geo_truth, geo_pred))
        acc_at_161 = len([d for d in grid_dists if d < 161]) / float(test)
        print('Acc@161 = {:.4f}'.format(acc_at_161))
        print('Grid-Mean = {}'.format(int(np.mean(grid_dists))))
        print('Grid-Median = {}'.format(int(np.median(grid_dists))))


if __name__ == '__main__':
    main()
