#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""convert the dumped file of geogcn (Rahimi et al., ACL'18) to the SSFGM format."""


from __future__ import (
    division, absolute_import, print_function, unicode_literals)
import gzip
import json
import argparse

import six
from six.moves import cPickle, map, xrange
import numpy as np
from scipy import sparse


def main():
    parser = argparse.ArgumentParser(
        description='convert geogcn format to SSFGM format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('vocab_path', help='vocabulary path')
    parser.add_argument('data_path', help='data path')
    parser.add_argument('out_path', help='output path')
    parser.add_argument('grid_path', help='save grid median')
    parser.add_argument('user_loc', help='save user location')
    args = parser.parse_args()

    with gzip.open(args.vocab_path, 'rb') as fin:
        vocab2id = cPickle.load(fin, encoding='latin1')
    print(len(vocab2id))

    with gzip.open(args.data_path, 'rb') as fin:
        data = cPickle.load(fin, encoding='latin1')
    print('Data is loaded from {}.'.format(args.data_path))
    (adj_, x_train, y_train, x_dev, y_dev, x_test, y_test, u_train, u_dev,
        u_test, lat_median, lon_median, user_loc_) = data

    grid_median = {}
    for i in lat_median:
        grid_median[i] = [float(lat_median[i]), float(lon_median[i])]
    with open(args.grid_path, 'wt', encoding='utf-8') as fout:
        json.dump(grid_median, fout)

    # restore adj
    new_data = np.ones_like(adj_.data, dtype=np.int64)
    adj = sparse.csr_matrix(
        (new_data, adj_.indices.copy(), adj_.indptr.copy()),
        shape=adj_.shape, dtype=new_data.dtype)
    adj.setdiag(0.)
    adj.eliminate_zeros()

    # concat features, labels
    features = sparse.vstack([x_train, x_dev, x_test])
    if len(y_train.shape) == 1:
        labels = np.hstack((y_train, y_dev, y_test))
    else:
        raise NotImplementedError()
    features = features.astype(np.float32)
    labels = labels.astype(np.int32)
    print(adj.nnz, adj.shape, features.shape, labels.shape)

    # train_index, val_index, test_index
    idx = -1
    user2idx = {}
    train_index, val_index, test_index = [], [], []
    for user in u_train:
        idx += 1
        train_index.append(idx)
        user2idx[user] = idx
    for user in u_dev:
        idx += 1
        val_index.append(idx)
        user2idx[user] = idx
    for user in u_test:
        idx += 1
        test_index.append(idx)
        user2idx[user] = idx

    n_labels = np.max(labels) + 1
    label2id = {str(i):i for i in xrange(n_labels)}
    obj = [adj, features, labels, train_index, val_index, test_index,
           vocab2id, label2id]
    with open(args.out_path, 'wb') as fout:
        cPickle.dump(obj, fout, protocol=4)

    user_loc = {}
    for user, loc in six.iteritems(user_loc_):
        user_loc[user2idx[user]] = list(map(float, loc.split(',')))
    with open(args.user_loc, 'wt', encoding='utf-8') as fout:
        json.dump(user_loc, fout)


if __name__ == '__main__':
    main()
