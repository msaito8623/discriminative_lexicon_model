"""
Path weaving: ordering predicted form cues into word forms by
synthesis-by-analysis, following Heitmeier, Chuang, & Baayen (2026) The
Discriminative Lexicon: Theory, Implementation in the Julia Package JudiLing,
and Applications.
"""

import numpy as np
import xarray as xr

from .mapping import to_cues, to_ngram, infer_gram

__all__ = ['cue_sequence', 'gen_amat', 'gen_mmats', 'gen_ymats', 'gen_yhats']

def cue_sequence (word, gram=3):
    """
    Returns the ordered cues of a word. Duplicates are kept. This is merely a
    wrapper of mapping.to_ngram.
    """
    return to_ngram(word, gram=gram, unique=False)

def gen_amat (words, gram=3, cues=None):
    """
    Constructs an adjacency matrix (A-matrix) of attested cue transitions,
    following Heitmeier, Chuang, & Baayen (2026). The A matrix is the same as
    the V matrix produced by mapping.gen_vmat, except that the V matrix
    considers every n-gram sequences, regardless of a training dataset, while
    the A matrix considers every transition of n-grams from the words present
    in the training dataset.

    Returns
    -------
    amat : xarray.DataArray
        A boolean matrix with the dimensions (current, next).
    """
    if cues is None:
        cues = to_cues(words, gram=gram)
    gram = infer_gram(cues)
    index = {c: i for i, c in enumerate(cues)}
    cuelen = len(cues)
    amat = np.zeros((cuelen, cuelen), dtype=bool)
    for word in words:
        seq = cue_sequence(word, gram=gram)
        for cur, nex in zip(seq[:-1], seq[1:]):
            if (cur in index) and (nex in index):
                amat[index[cur], index[nex]] = True
    amat = xr.DataArray(amat, dims=('current', 'next'),
                        coords={'current': cues, 'next': cues})
    return amat

def gen_ymats (cmat, gram=None):
    """
    Generates the positional form matrices Y_1 ... Y_n from a C-matrix,
    following Heitmeier, Chuang, & Baayen (2018). In Y_n, the cue occupying
    position n of a word is marked as 1 and 0 elsewhere. The number of
    matrices, i.e. the max of n, is the cue count of the longest word.

    Returns
    -------
    ymats : list of xarray.DataArray
        Each with the dimensions and coordinates of cmat.
    """
    words = list(cmat[cmat.dims[0]].values)
    cues = list(cmat[cmat.dims[1]].values)
    if gram is None:
        gram = infer_gram(cues)
    index = {c: i for i, c in enumerate(cues)}
    seqs = [cue_sequence(w, gram=gram) for w in words]
    ymats = []
    words_lens = [ len(s) for s in seqs ]
    max_word_len = max(words_lens)
    for pos in range(max_word_len): # For each position from 0 ... n
        ymat = np.zeros(cmat.shape, dtype=float)
        for i, seq in enumerate(seqs): # For each word.
            if pos < len(seq) and seq[pos] in index:
                ymat[i, index[seq[pos]]] = 1.0
        ymat = xr.DataArray(ymat, dims=cmat.dims, coords=cmat.coords)
        ymats.append(ymat)
    return ymats

def gen_mmats (cmat, ymats):
    """
    Estimate the positional mapping matrices M_n.

    The position matrices Y_1, Y_2, ..., Y_n serve as the gold-standard signal.
    Based on C as input and these position matrices as output, the positional
    mapping matrices M_1, M_2, ..., M_n are estimated, namely CM_n = Y_n.

    Though there are a total of n matrices to estimate, all of them share the
    core (C^T C)^-1 C^T, and it is enough to be inverted and estimated once.
    See Heitmeier et al. (2026) for more details.

    Returns
    -------
    mmats : list of xarray.DataArray
        Each with dimensions (cues, cues).
    """
    def _gen_mmats (y, core, cues):
        m = xr.DataArray(np.matmul(core, np.array(y)), dims=('cues', 'cues_to'),
                     coords={'cues': cues, 'cues_to': cues.values})
        return m
    cues = cmat[cmat.dims[1]]
    c = np.array(cmat)
    core = np.matmul(np.linalg.pinv(np.matmul(c.T, c)), c.T)
    mmats = [_gen_mmats(y, core, cues) for y in ymats]
    return mmats

def gen_yhats (chat, mmats):
    """
    Predict the positional support matrices Y_n from a predicted C-matrix
    (i.e., a C-hat matrix).

    Returns
    -------
    yhats : list of xarray.DataArray
        Each with the dimensions and coordinates of chat.
    """
    def _gen_yhats (m, chat):
        yh = xr.DataArray(np.matmul(chat.values, np.array(m)), dims=chat.dims,
                          coords=chat.coords)
        return yh
    yhats = [_gen_yhats(m, chat) for m in mmats]
    return yhats
