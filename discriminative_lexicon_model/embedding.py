from itertools import accumulate, groupby
from operator import itemgetter

import numpy as np
import xarray as xr

__all__ = [
    "sentence_to_ngrams",
    "corpus_to_ngrams",
    "corpus_to_words",
    "gen_form_vectors",
    "iter_form_vectors",
    "predictive_learning",
    "extract_word_vectors",
    "prediction_errors",
    "save_model",
    "load_model",
]

try:
    import torch
except ImportError:
    torch = None

try:
    import numba
except ImportError:
    numba = None

def sentence_to_ngrams (words, gram=3, boundary='#'):
    """
    Convert one sentence into a sequence of n-grams.

    ['it', 'is', 'good'] -> '#it#is#good#' -> ['#it', 'it#', 't#i', ...].
    Used when continuous=False; see corpus_to_ngrams for the continuous case.

    Parameters
    ----------
    words : list of str
        The words of one sentence.
    gram : int
        N-gram size. Default is 3.
    boundary : str
        Word and sentence boundary character. Default is '#'.

    Returns
    -------
    ngrams : list of str
        The n-grams, in order, not deduplicated.
    """
    s = boundary + boundary.join(words) + boundary
    ngrams = [s[i:i+gram] for i in range(len(s) - gram + 1)]
    return ngrams

def corpus_to_ngrams (sentences, gram=3, boundary='#'):
    """
    Convert a whole corpus into one continuous sequence of n-grams.

    Words are separated by a single boundary character and sentences by a
    doubled one: [['you', 'go'], ['i', 'go']] -> '##you#go##i#go##'.

    Parameters
    ----------
    sentences : list of list of str
        The corpus. Each inner list is one sentence.
    gram : int
        N-gram size. Default is 3.
    boundary : str
        Word boundary character. Sentence boundaries are this character
        doubled. Default is '#'.

    Returns
    -------
    ngrams : list of str
        One continuous sequence of n-grams for the whole corpus.
    """
    sep = boundary * 2
    s = sep + sep.join(boundary.join(sent) for sent in sentences) + sep
    ngrams = [s[i:i+gram] for i in range(len(s) - gram + 1)]
    return ngrams

def corpus_to_words (sentences, boundary='#'):
    """
    Convert a whole corpus into one continuous sequence of word tokens.

    The doubled boundary character is an ordinary token, occupying one
    timestep and one slot in the window just as a word does:
    [['you', 'go'], ['i', 'go']] -> ['##', 'you', 'go', '##', 'i', 'go', '##'].

    Parameters
    ----------
    sentences : list of list of str
        The corpus. Each inner list is one sentence.
    boundary : str
        Word boundary character. The sentence boundary token is this
        character doubled. Default is '#'.

    Returns
    -------
    tokens : list of str
        One continuous sequence of tokens for the whole corpus.
    """
    sep = boundary * 2
    tokens = [tok for sent in sentences
              for tok in [sep] + list(sent)] + [sep]
    return tokens

def iter_form_vectors (sequence, cue_index, window=4):
    """
    Yield the form vectors c_t of a sequence one timestep at a time.

    Identical in output to gen_form_vectors, but holds only one timestep
    in memory.

    Parameters
    ----------
    sequence : list of str
        A sequence of items (n-grams or words).
    cue_index : dict
        Mapping from item string to column index in the form vector.
    window : int
        The sliding window size. Default is 4.

    Yields
    ------
    c_t : numpy.ndarray
        Shape (n_cues,), one per timestep.
        There are len(sequence) + window + 1 of them.
    """
    n_cues = len(cue_index)
    for idx, val in _iter_active_cues(sequence, cue_index, window=window):
        c_t = np.zeros(n_cues)
        c_t[idx] = val
        yield c_t

def _iter_active_cues (sequence, cue_index, window=4, step=1):
    """Yield (indices, values) for the active cues of each form vector.

    Keeps the cost per timestep independent of vocabulary size; `step`
    corresponds to `update_every` in predictive_learning.
    """
    n_timesteps = len(sequence) + window + 1
    decay_step = 1.0 / window
    n_seq = len(sequence)

    for t in range(0, n_timesteps, step):
        # Descending offset means the most recent occurrence is written
        # last, so it wins when a cue recurs inside the window.
        active = {}
        for offset in range(window - 1, -1, -1):
            pos = t - 1 - offset
            if 0 <= pos < n_seq:
                active[cue_index[sequence[pos]]] = 1.0 - offset * decay_step
        n = len(active)
        yield (np.fromiter(active.keys(), dtype=np.intp, count=n),
               np.fromiter(active.values(), dtype=float, count=n))

def gen_form_vectors (sequence, cue_index, window=4):
    """
    Generate form vectors c_t for a sequence of items with linear decay.

    Items inside the sliding window are activated at 1 - offset/window,
    with a leading zero vector and trailing fade-out steps.

    Parameters
    ----------
    sequence : list of str
        A sequence of items (n-grams or words).
    cue_index : dict
        Mapping from item string to column index in the form vector.
    window : int
        The sliding window size. Default is 4.

    Returns
    -------
    form_vectors : numpy.ndarray
        Shape (n_timesteps, n_cues).
        n_timesteps = len(sequence) + window + 1.
    """
    form_vectors = np.array(list(iter_form_vectors(sequence, cue_index,
                                                   window=window)))
    return form_vectors

def _word_timesteps (sentence, gram):
    """Timestep at which each word of one sentence is fully perceived.

    The per-sentence counterpart of _corpus_word_timesteps.
    """
    trailing = list(accumulate((1 + len(w) for w in sentence), initial=0))[1:]
    timesteps = [pos - gram + 2 for pos in trailing]
    return timesteps

def _corpus_word_timesteps (sentences, mode, gram, boundary='#'):
    """Timestep at which each word of the corpus is fully perceived.

    That is, when the word's last n-gram enters the window. One timestep
    per word occurrence, in corpus order, flattened across sentences.
    """
    sep = len(boundary) * 2
    timesteps = []

    if mode == 'word':
        # corpus_to_words puts one boundary token before each sentence
        # and one at the end; an item at sequence position p peaks at t = p + 1.
        pos = 1
        for sentence in sentences:
            for _ in sentence:
                timesteps.append(pos + 1)
                pos += 1
            pos += 1                      # the boundary token
    else:
        idx = sep
        for sentence in sentences:
            for i, word in enumerate(sentence):
                trailing = idx + len(word)
                timesteps.append(trailing - gram + 2)
                idx = trailing + (1 if i < len(sentence) - 1 else sep)

    return timesteps

def predictive_learning (sentences, semantic_dim=100,
                         lr_f=0.01, lr_w=0.01, window=4,
                         gram=3, epochs=1, randseed=None,
                         backend='auto', device=None,
                         mode='trigram', continuous=True, update_every=1):
    """
    Self-supervised predictive learning of semantic representations.

    Learns two matrices from word sequences alone, with the model's own
    prediction error as the learning signal and no target semantics given:
    F (n_form x semantic_dim), mapping form vectors to semantic vectors,
    and W (semantic_dim x semantic_dim), predicting the next semantic
    vector from the current one.

    See the "Predictive learning" page of the documentation for the update
    rule and the handling of sentence boundaries.

    Parameters
    ----------
    sentences : list of list of str
        Each inner list is a sentence (sequence of words). Sentences are
        taken to be in order and are processed as one stream.
    semantic_dim : int
        Dimensionality of the emergent semantic space. Default 100.
    lr_f : float
        Learning rate for F matrix updates. Default 0.01.
    lr_w : float
        Learning rate for W matrix updates. Default 0.01.
    window : int
        Sliding window size for form vectors. Default 4.
    gram : int
        N-gram size. Only used when mode='trigram'. Default is 3.
    epochs : int
        Number of passes over the sentence list. Default 1.
    randseed : int or None
        Random seed for reproducibility of weight initialization.
    backend : {'auto', 'numpy', 'numba', 'torch'}
        Default 'auto', which is numba if installed and NumPy otherwise.
        'numpy' is the reference implementation and 'numba' the same rule
        compiled single-threaded, much the fastest at moderate
        semantic_dim. 'torch' overtakes numba above semantic_dim of roughly
        500; see the documentation.
    device : str or None
        For the torch backend: 'cuda', 'cpu', etc. If None, 'cuda' when
        available, else 'cpu'. A GPU only helps at large semantic_dim.
    mode : {'trigram', 'word'}
        'trigram' -> the corpus becomes one n-gram sequence; F dims are
            ('cues', 'semantics').
        'word' -> words are the input sequence directly, with '##' as an
            ordinary token; F dims are ('word', 'semantics').
    continuous : bool
        Default True: the corpus is one stream and sentence boundaries are
        the ordinary cue '##'. False processes each sentence on its own
        timeline, resetting the semantic context at every boundary, which
        suits unordered or unrelated sentences.
    update_every : int
        Update the weights every update_every-th timestep instead of every
        timestep. Default 1.

    Returns
    -------
    result : dict
        'F' : xarray.DataArray, shape (n_form, semantic_dim)
        'W' : xarray.DataArray, shape (semantic_dim, semantic_dim)
    """
    if mode == 'trigram':
        sequences = ([corpus_to_ngrams(sentences, gram=gram)] if continuous
                     else [sentence_to_ngrams(s, gram=gram) for s in sentences])
        form_dim_name = 'cues'
    elif mode == 'word':
        sequences = ([corpus_to_words(sentences)] if continuous
                     else [list(s) for s in sentences])
        form_dim_name = 'word'
    else:
        raise ValueError(f'Unknown mode "{mode}". Use "trigram" or "word".')

    form_list = list(dict.fromkeys(item for s in sequences for item in s))

    form_index = {item: i for i, item in enumerate(form_list)}
    n_form = len(form_list)
    sem_labels = ['S{:03d}'.format(i) for i in range(semantic_dim)]

    rng = np.random.default_rng(randseed)
    F_init = rng.normal(scale=0.01, size=(n_form, semantic_dim))
    W_init = rng.normal(scale=0.01, size=(semantic_dim, semantic_dim))

    # Determine backend
    if backend == 'auto':
        backend = 'numba' if numba is not None else 'numpy'
    if backend == 'torch':
        if torch is None:
            raise ImportError('PyTorch is not installed. Install it to use the "torch" backend.')
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif backend == 'numba':
        if numba is None:
            raise ImportError('numba is not installed. Install it to use the "numba" backend.')
    elif backend != 'numpy':
        raise ValueError(f'Unknown backend "{backend}". '
                         'Use "numpy", "numba", "torch", or "auto".')

    if backend == 'torch':
        F, W = _predictive_learning_torch(
            sequences, form_index, F_init, W_init,
            semantic_dim, lr_f, lr_w, window, epochs, device, update_every)
    elif backend == 'numba':
        F, W = _predictive_learning_numba(
            sequences, form_index, F_init, W_init,
            semantic_dim, lr_f, lr_w, window, epochs, update_every)
    else:
        F, W = _predictive_learning_numpy(
            sequences, form_index, F_init, W_init,
            semantic_dim, lr_f, lr_w, window, epochs, update_every)

    F_xr = xr.DataArray(F, dims=(form_dim_name, 'semantics'),
                         coords={form_dim_name: form_list,
                                 'semantics': sem_labels})
    W_xr = xr.DataArray(W, dims=('semantics_from', 'semantics_to'),
                         coords={'semantics_from': sem_labels,
                                 'semantics_to': sem_labels})
    return {'F': F_xr, 'W': W_xr}

def extract_word_vectors (sentences, F, gram=3, window=4, continuous=True):
    """
    Extract word-level semantic vectors from a trained trigram F matrix.

    Runs the forward pass and reads each word off at the timestep where
    its last n-gram enters the window, averaging over occurrences.

    Parameters
    ----------
    sentences : list of list of str
        The corpus (same format as for predictive_learning).
    F : xarray.DataArray
        The trained F matrix with dims ('cues', 'semantics').
    gram : int
        N-gram size. Must match what was used for training. Default 3.
    window : int
        Sliding window size. Must match what was used for training.
        Default 4.
    continuous : bool
        Must match the value used for training. Default True. See
        predictive_learning.

    Returns
    -------
    word_vectors : xarray.DataArray
        Shape (n_words, semantic_dim), dims ('word', 'semantics').
        Words are sorted alphabetically.
    """
    cue_index = {c: i for i, c in enumerate(F.cues.values)}
    F_vals = F.values

    if continuous:
        pieces = [(corpus_to_ngrams(sentences, gram=gram),
                   _corpus_word_timesteps(sentences, 'trigram', gram),
                   [w for sentence in sentences for w in sentence])]
    else:
        pieces = [(sentence_to_ngrams(sentence, gram=gram),
                   _word_timesteps(sentence, gram), sentence)
                  for sentence in sentences]

    def pairs ():
        """(word, embedding) for every word occurrence."""
        for seq, timesteps, words in pieces:
            wanted = set(timesteps)
            # Stream the form vectors and keep only the ones we are asked for.
            vecs = {t: val @ F_vals[idx]
                    for t, (idx, val) in enumerate(
                        _iter_active_cues(seq, cue_index, window=window))
                    if t in wanted}
            for word, t in zip(words, timesteps):
                if t in vecs:
                    yield word, vecs[t]

    occurrences = sorted(pairs(), key=itemgetter(0))
    grouped = [(word, [vec for _, vec in group])
               for word, group in groupby(occurrences, key=itemgetter(0))]

    word_vectors = xr.DataArray(
        np.array([sum(vecs) / len(vecs) for _, vecs in grouped]),
        dims=('word', 'semantics'),
        coords={'word': [word for word, _ in grouped],
                'semantics': list(F.semantics.values)})
    return word_vectors

def prediction_errors (sentences, F, W, mode='word', gram=3, window=4,
                       metric='euclidean', continuous=True, update_every=1):
    """
    Compute prediction error for each word in each sentence.

    A forward pass through the trained model, with no weight updates. In
    word mode each timestep is one word; in trigram mode the error is
    read at the timestep where the word is fully perceived, as in
    extract_word_vectors.

    Parameters
    ----------
    sentences : list of list of str
        The corpus.
    F : xarray.DataArray
        Trained F matrix.
    W : xarray.DataArray
        Trained W matrix.
    mode : {'word', 'trigram'}
        Must match the mode used for training.
    gram : int
        N-gram size. Only used when mode='trigram'. Default 3.
    window : int
        Sliding window size. Must match training. Default 4.
    metric : {'euclidean', 'cosine'}
        How to measure the prediction error. Both are distances, so larger
        means a worse prediction.
        'euclidean' -> norm(s_hat - m_hat)
        'cosine'    -> 1 - cos(s_hat, m_hat), in [0, 2]
    continuous : bool
        Must match the value used for training. Default True. See
        predictive_learning.
    update_every : int
        Must match the value used for training. Default 1. A word is
        scored at the last sampled timestep at or before the one where it
        is fully perceived.

    Returns
    -------
    results : list of list of (str, float)
        For each sentence, a list of (word, error) pairs.
    """
    if mode == 'word':
        form_dim = 'word'
    else:
        form_dim = 'cues'

    form_list = list(F[form_dim].values)
    form_index = {item: i for i, item in enumerate(form_list)}
    F_vals = F.values
    W_vals = W.values
    semantic_dim = F_vals.shape[1]

    def _compute_error(s_hat_t, m_hat_t):
        if metric == 'euclidean':
            distance = np.linalg.norm(s_hat_t - m_hat_t)
            return distance
        else:  # cosine
            norm_s = np.linalg.norm(s_hat_t)
            norm_m = np.linalg.norm(m_hat_t)
            if norm_s == 0 or norm_m == 0:
                # Cosine is undefined against a zero vector. 1.0 is the
                # orthogonal value, i.e. no information -- the midpoint of
                # the [0, 2] range, not its maximum.
                return 1.0
            similarity = np.dot(s_hat_t, m_hat_t) / (norm_s * norm_m)
            distance = 1.0 - similarity
            return distance

    def errors_for (seq, timesteps):
        """(timestep -> error) for one sequence.

        s_prev advances at every timestep, but an error is only recorded
        where a word is fully perceived. It starts at zero, which is the
        context reset at the head of the sequence.
        """
        # With update_every > 1 only every update_every-th timestep exists,
        # so a word is scored at the last sampled step at or before its own.
        wanted = {t - t % update_every for t in timesteps}
        errors_at_t = {}
        s_prev = np.zeros(semantic_dim)
        for i, (idx, val) in enumerate(_iter_active_cues(seq, form_index,
                                                         window=window,
                                                         step=update_every)):
            t = i * update_every
            s_hat = val @ F_vals[idx]
            if t in wanted:
                errors_at_t[t] = _compute_error(s_hat, s_prev @ W_vals)
            s_prev = s_hat
        return errors_at_t

    if continuous:
        seq = (corpus_to_ngrams(sentences, gram=gram) if mode == 'trigram'
               else corpus_to_words(sentences))
        timesteps = _corpus_word_timesteps(sentences, mode, gram)
        errors_at_t = errors_for(seq, timesteps)
        # Split the flat corpus-order list back into sentences.
        out, start = [], 0
        for sentence in sentences:
            out.append([(word, errors_at_t[t - t % update_every])
                        for word, t in zip(sentence,
                                           timesteps[start:start + len(sentence)])
                        if t - t % update_every in errors_at_t])
            start += len(sentence)
        return out

    out = []
    for sentence in sentences:
        seq = (sentence_to_ngrams(sentence, gram=gram) if mode == 'trigram'
               else list(sentence))
        timesteps = ([i + 1 for i, _ in enumerate(sentence)] if mode == 'word'
                     else _word_timesteps(sentence, gram))
        errors_at_t = errors_for(seq, timesteps)
        out.append([(word, errors_at_t[t - t % update_every])
                    for word, t in zip(sentence, timesteps)
                    if t - t % update_every in errors_at_t])
    return out


def save_model (result, path):
    """
    Save a trained model (F and W matrices) to a NetCDF file.

    Parameters
    ----------
    result : dict
        The dict returned by predictive_learning, containing
        'F' and 'W' as xarray.DataArrays.
    path : str
        File path to save to (e.g., 'model_word.nc').
    """
    ds = xr.Dataset({'F': result['F'], 'W': result['W']})
    ds.to_netcdf(path)


def load_model (path):
    """
    Load a trained model from a NetCDF file.

    Parameters
    ----------
    path : str
        File path to load from.

    Returns
    -------
    result : dict
        Dict with 'F' and 'W' as xarray.DataArrays.
    """
    ds = xr.open_dataset(path)
    result = {'F': ds['F'], 'W': ds['W']}
    ds.close()
    return result


def _predictive_learning_numpy (sequences, form_index, F, W,
                                semantic_dim, lr_f, lr_w, window, epochs,
                                update_every=1):
    n_form = F.shape[0]

    def update_step (state, idx, val):
        """One timestep of the update rule.

        (idx, val) is the sparse c_t: only rows idx of F are read, and only
        rows prev_idx of F are written, since every other entry of c is zero.
        """
        F, W, s_prev, prev_idx, prev_val = state
        s_hat_t = val @ F[idx]
        m_hat_t = s_prev @ W
        e_t = s_hat_t - m_hat_t

        delta_t = e_t @ W.T

        W += lr_w * (s_prev[:, None] * e_t)
        F[prev_idx] += lr_f * prev_val[:, None] * delta_t

        return F, W, s_hat_t, idx, val

    def learn_sequence (F, W, sequence):
        """One sequence, with the semantic context reset at its start.

        Continuous mode passes a single sequence (the whole corpus);
        per-sentence mode passes one per sentence. The form vectors are
        streamed rather than materialised, since in continuous mode the
        dense (n_timesteps, n_cues) array would not fit in memory.
        """
        empty = np.empty(0, dtype=np.intp), np.empty(0)
        state = (F, W, np.zeros(semantic_dim), *empty)
        for idx, val in _iter_active_cues(sequence, form_index, window=window,
                                          step=update_every):
            state = update_step(state, idx, val)
        F, W = state[0], state[1]
        return F, W

    F, W = F.copy(), W.copy()
    for _ in range(epochs):
        for sequence in sequences:
            F, W = learn_sequence(F, W, sequence)

    return F, W

def _numba_train (seq_idx, bounds, F, W, lr_f, lr_w, window, epochs, step):
    """All epochs of the update rule, written for numba to compile."""
    dim = F.shape[1]
    decay = 1.0 / window
    n_sequences = len(bounds) - 1

    s_prev = np.zeros(dim)
    s_hat = np.zeros(dim)
    m_hat = np.zeros(dim)
    e = np.zeros(dim)
    d = np.zeros(dim)
    cur_i = np.empty(window, np.int64)
    cur_v = np.empty(window)
    pre_i = np.empty(window, np.int64)
    pre_v = np.empty(window)

    for _ in range(epochs):
        for s in range(n_sequences):
          start = bounds[s]
          n_seq = bounds[s + 1] - start
          n_steps = n_seq + window + 1

          for j in range(dim):
              s_prev[j] = 0.0
          n_pre = 0

          for t in range(0, n_steps, step):
              n_cur = 0
              for offset in range(window - 1, -1, -1):
                  pos = t - 1 - offset
                  if 0 <= pos < n_seq:
                      cue = seq_idx[start + pos]
                      activation = 1.0 - offset * decay
                      seen = -1
                      for k in range(n_cur):
                          if cur_i[k] == cue:
                              seen = k
                              break
                      if seen >= 0:
                          cur_v[seen] = activation
                      else:
                          cur_i[n_cur] = cue
                          cur_v[n_cur] = activation
                          n_cur += 1

              for j in range(dim):                       # s_hat_t = c_t F
                  total = 0.0
                  for k in range(n_cur):
                      total += cur_v[k] * F[cur_i[k], j]
                  s_hat[j] = total
              for j in range(dim):                       # m_hat_t = s_prev W
                  total = 0.0
                  for k in range(dim):
                      total += s_prev[k] * W[k, j]
                  m_hat[j] = total
              for j in range(dim):                       # e_t
                  e[j] = s_hat[j] - m_hat[j]
              for j in range(dim):                       # delta_t = e_t W^T
                  total = 0.0
                  for k in range(dim):
                      total += e[k] * W[j, k]
                  d[j] = total
              for i in range(dim):                       # W += lr_w s_prev^T e
                  scale = lr_w * s_prev[i]
                  for j in range(dim):
                      W[i, j] += scale * e[j]
              for k in range(n_pre):                     # F += lr_f c_prev^T delta
                  row = pre_i[k]
                  scale = lr_f * pre_v[k]
                  for j in range(dim):
                      F[row, j] += scale * d[j]

              for j in range(dim):
                  s_prev[j] = s_hat[j]
              n_pre = n_cur
              for k in range(n_cur):
                  pre_i[k] = cur_i[k]
                  pre_v[k] = cur_v[k]

    return F, W


_numba_train_compiled = None


def _predictive_learning_numba (sequences, form_index, F, W,
                                semantic_dim, lr_f, lr_w, window, epochs,
                                update_every=1):
    global _numba_train_compiled
    if _numba_train_compiled is None:
        _numba_train_compiled = numba.njit(cache=True)(_numba_train)

    flat = [form_index[item] for sequence in sequences for item in sequence]
    seq_idx = np.fromiter(flat, dtype=np.int64, count=len(flat))
    bounds = np.cumsum([0] + [len(s) for s in sequences]).astype(np.int64)
    F, W = _numba_train_compiled(seq_idx, bounds, F.copy(), W.copy(),
                                 lr_f, lr_w, window, epochs, update_every)
    return F, W


def _predictive_learning_torch (sequences, form_index, F_init, W_init,
                                semantic_dim, lr_f, lr_w, window, epochs,
                                device, update_every=1):
    n_form = F_init.shape[0]

    def zeros (n):
        return torch.zeros(n, dtype=torch.float32, device=device)

    def update_step (state, idx, val):
        """One timestep of the update rule.

        (idx, val) is the sparse c_t: only rows idx of F are read, and only
        rows prev_idx of F are written, since every other entry of c is zero.
        """
        F, W, s_prev, prev_idx, prev_val = state
        s_hat_t = val @ F[idx]
        m_hat_t = s_prev @ W
        e_t = s_hat_t - m_hat_t

        delta_t = e_t @ W.T

        W += lr_w * (s_prev[:, None] * e_t)
        F[prev_idx] += lr_f * prev_val[:, None] * delta_t

        return F, W, s_hat_t, idx, val

    def learn_sequence (F, W, sequence):
        """One sequence, with the semantic context reset at its start."""
        empty = (torch.empty(0, dtype=torch.long, device=device),
                 torch.empty(0, dtype=torch.float32, device=device))
        state = (F, W, zeros(semantic_dim), *empty)
        for idx, val in _iter_active_cues(sequence, form_index, window=window,
                                          step=update_every):
            state = update_step(
                state,
                torch.as_tensor(idx, dtype=torch.long, device=device),
                torch.as_tensor(val, dtype=torch.float32, device=device))
        F, W = state[0], state[1]
        return F, W

    F = torch.tensor(F_init, dtype=torch.float32, device=device)
    W = torch.tensor(W_init, dtype=torch.float32, device=device)
    for _ in range(epochs):
        for sequence in sequences:
            F, W = learn_sequence(F, W, sequence)

    F_out, W_out = F.cpu().numpy(), W.cpu().numpy()
    return F_out, W_out
