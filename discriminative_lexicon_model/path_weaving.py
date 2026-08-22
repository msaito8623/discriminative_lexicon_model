"""
Path weaving: ordering predicted form cues into word forms by
synthesis-by-analysis, following Heitmeier, Chuang, & Baayen (2026) The
Discriminative Lexicon: Theory, Implementation in the Julia Package JudiLing,
and Applications.
"""

import numpy as np
import xarray as xr
from .mapping import to_cues, to_ngram, infer_gram, gen_cmat, _concat_selected

__all__ = ['cue_sequence', 'gen_amat', 'gen_mmats', 'gen_ymats', 'gen_yhats',
           'find_paths', 'select_path', 'weave']

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
    following Heitmeier, Chuang, & Baayen (2026). In Y_n, the cue occupying
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

def _row (mat, word=None):
    """
    A utility function for an internal use. It returns one row of a matrix as a
    numpy array, which can be selected by word or index.
    """
    arr = mat if isinstance(mat, np.ndarray) else np.array(mat)
    if arr.ndim == 1:
        row = arr
    elif (word is None) and (arr.shape[0] == 1):
        row = arr[0]
    elif word is None:
        raise ValueError('"word" is missing. "word" can be omitted only when "arr" is already a single row vector.')
    elif isinstance(mat, xr.DataArray) and not isinstance(word, (int, np.integer)):
        row = mat.sel({mat.dims[0]: word}).values
    else:
        row = arr[word]
    return row

def find_paths (chat, yhats, vmat, word=None, amat=None, threshold=0.1, max_paths=None):
    """
    Find possible paths for a given word.

    This method identifies possible paths at each step/position. For each
    step/position, next possible cues are selected, based on activations in the
    C-hat vector of question (i.e., is the cue supported by the word meaning
    strongly enough?), activations in the Y-hat of the position (i.e., is the
    cue motivated strongly enough by the word meaning through its predicted
    forms, given the position?), and the last cue (i.e., the next cue must be
    legitimate continuations of the last cue). Continuation eligibility is
    checked by the A-matrix, which permits only the transitions of cues that
    are attested in the training data, which avoids non-word sequences.

    The paths that are still growing are named "growing" and the paths that
    reached the end (i.e., a cue with "#" at its end) get "harvested".

    Returns
    -------
    paths : list of tuple of str
        Each path containing a sequence of cues
    """
    cues = list(vmat['next'].values)
    ch_row = _row(chat, word)
    yh_rows = [_row(y, word) for y in yhats]
    if amat is None:
        v = vmat.sel(current=cues).values # Drop the initial-position row, i.e. ""
    else:
        v = amat.sel(current=cues, next=cues).values
    ok = np.array(ch_row) > threshold
    yh_rows = [y > threshold for y in yh_rows]
    yh_rows = [ok & y for y in yh_rows]
    candidates = [np.where(y)[0] for y in yh_rows]
    init_cue = [i for i in candidates[0] if cues[i].startswith('#')] if candidates else []
    def _is_final (path):
        return len(path) > 1 and cues[path[-1]].endswith('#')
    def _finished (paths):
        return [ p for p in paths if _is_final(p) ]
    def _grow (path, pos):
        last_cue = path[-1]
        next_possible_cues = candidates[pos] # regardless what has been chosen so far.
        next_permitted_cues = [ j for j in next_possible_cues if v[last_cue,j] ]
        updated_paths = [ path+(j,) for j in next_permitted_cues ]
        return updated_paths
    def _extend (paths, pos):
        ongoing = [ p for p in paths if not _is_final(p) ]
        grown = []
        for path in ongoing: # For each candidate path.
            grown = grown + _grow(path, pos)
        return grown
    growing = [ (i,) for i in init_cue ] # Paths that are still growing.
    harvested = _finished(growing)
    for pos in range(1, len(candidates)):
        growing = _extend(growing, pos)
        harvested = harvested + _finished(growing)
        if (max_paths is not None) and (len(harvested) >= max_paths):
            break
    harvested = [ tuple(cues[i] for i in p) for p in harvested ]
    return harvested

def select_path (paths, gold, cmat, fmat, gram=None):
    """
    Synthesis-by-analysis: Decides on the winner form/path, given a set of
    paths.

    Each path gets concatenated to be a single form, which then gets mapped
    back onto semantics. The generated semantics then gets compared to the
    gold-standard vector. The path/form that results in the highest correlation
    with the gold-standard vector in terms of semantics will be the winner.

    Returns
    -------
    winner : str
        The winning form, or '' when paths is empty.
    corrs : dict
        The correlations of the candidate forms with gold in semantics.
    """
    if len(paths) == 0:
        return '', {}
    cues = list(cmat[cmat.dims[1]].values)
    if gram is None:
        gram = infer_gram(cues)
    forms = [_concat_selected(p, overlap=True).strip('#') for p in paths]
    forms = list(dict.fromkeys(forms))
    cmat_cand = gen_cmat(forms, gram=gram, cues=cues)
    shat_cand = cmat_cand @ fmat
    gold = np.array(gold).ravel()
    corrs = {f: float(np.corrcoef(s, gold)[0, 1]) for f, s in zip(forms, shat_cand)}
    valid = {f: c for f, c in corrs.items() if not np.isnan(c)}
    if len(valid) == 0:
        winner = ''
    else:
        winner = max(valid, key=valid.get)
    return winner, corrs

def weave (gold, cmat, fmat, chat, yhats, vmat, word=None, amat=None,
           threshold=0.1, gram=None):
    """
    A wrapper of find_paths and select_path: Produces the winner form, based on
    the the path-weaving algorithm (Heitmeier et al., 2026).

    Returns
    -------
    form : str
        The winner form. The empry string when no path is found.
    corrs : dict
        The correlations of all candidate forms with the gold-standard semantic
        vector.
    """
    paths = find_paths(chat, yhats, vmat, word=word, amat=amat, threshold=threshold)
    form, corrs = select_path(paths, gold, cmat, fmat, gram=gram)
    return form, corrs
