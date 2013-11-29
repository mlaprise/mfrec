'''
    Datasets Utility

    Created on October 20th, 2010

    @author: Martin Laprise

'''

import numpy as np


def create_bool_sparse_row(sparse_matrix):
    '''
    Create a row-based dense representation of boolean sparse matrix
    '''
    rows = sparse_matrix.nonzero()[0].astype(np.int32)
    cols = sparse_matrix.nonzero()[1].astype(np.int32)
    count = np.bincount(rows).astype(np.int32)

    return np.r_[0,count], cols


def create_bool_sparse_col(sparse_matrix):
    '''
    Create a column-based dense representation of boolean sparse matrix
    '''
    rows = sparse_matrix.T.nonzero()[0].astype(np.int32)
    cols = sparse_matrix.T.nonzero()[1].astype(np.int32)
    count = np.bincount(rows).astype(np.int32)

    return np.r_[0,count], cols

