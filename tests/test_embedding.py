import numpy as np
import pytest
import discriminative_lexicon_model as dlm
from discriminative_lexicon_model.embedding import (sentence_to_ngrams,
                                                    corpus_to_ngrams,
                                                    corpus_to_words,
                                                    gen_form_vectors,
                                                    iter_form_vectors,
                                                    predictive_learning,
                                                    prediction_errors,
                                                    extract_word_vectors,
                                                    save_model,
                                                    load_model,
                                                    _corpus_word_timesteps)

# torch and numba are optional extras, so the backends that need them are
# skipped rather than failed when they are absent.
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


class TestSentenceToNgrams:
    def test_basic (self):
        # '#it#is#good#'
        assert sentence_to_ngrams(['it', 'is', 'good'], gram=3) == [
            '#it', 'it#', 't#i', '#is', 'is#', 's#g', '#go', 'goo',
            'ood', 'od#']

    def test_single_word (self):
        assert sentence_to_ngrams(['cat'], gram=3) == ['#ca', 'cat', 'at#']

    def test_bigrams (self):
        assert sentence_to_ngrams(['hi'], gram=2) == ['#h', 'hi', 'i#']

    def test_crosses_word_boundary (self):
        assert 't#i' in sentence_to_ngrams(['it', 'is'], gram=3)

    def test_custom_boundary (self):
        assert sentence_to_ngrams(['ab'], gram=2, boundary='%') == [
            '%a', 'ab', 'b%']

    def test_single_boundary_unlike_corpus_to_ngrams (self):
        """One sentence on its own timeline: a single '#', never a doubled one."""
        assert sentence_to_ngrams(['a'], gram=2) == ['#a', 'a#']
        assert corpus_to_ngrams([['a']], gram=2) == ['##', '#a', 'a#', '##']


class TestCorpusToNgrams:
    def test_basic (self):
        # '##it#is#good##'
        result = corpus_to_ngrams([['it', 'is', 'good']], gram=3)
        expected = ['##i', '#it', 'it#', 't#i', '#is', 'is#', 's#g',
                    '#go', 'goo', 'ood', 'od#', 'd##']
        assert result == expected

    def test_two_sentences (self):
        # '##you#go##i#go##'
        assert corpus_to_ngrams([['you', 'go'], ['i', 'go']], gram=3) == [
            '##y', '#yo', 'you', 'ou#', 'u#g', '#go', 'go#', 'o##',
            '##i', '#i#', 'i#g', '#go', 'go#', 'o##']

    def test_single_word (self):
        # '##cat##'
        assert corpus_to_ngrams([['cat']], gram=3) == ['##c', '#ca', 'cat',
                                                        'at#', 't##']

    def test_bigrams (self):
        # '##hi##'
        assert corpus_to_ngrams([['hi']], gram=2) == ['##', '#h', 'hi',
                                                       'i#', '##']

    def test_preserves_duplicates (self):
        result = corpus_to_ngrams([['a', 'a']], gram=3)
        assert result.count('#a#') == 2

    def test_crosses_word_boundary (self):
        assert 't#i' in corpus_to_ngrams([['it', 'is']], gram=3)

    def test_crosses_sentence_boundary (self):
        """The whole point: n-grams span the sentence boundary."""
        result = corpus_to_ngrams([['it'], ['is']], gram=3)
        assert 't##' in result and '##i' in result

    def test_custom_boundary (self):
        assert corpus_to_ngrams([['ab']], gram=2, boundary='%') == [
            '%%', '%a', 'ab', 'b%', '%%']


class TestCorpusToWords:
    def test_basic (self):
        assert corpus_to_words([['you', 'go'], ['i', 'go']]) == [
            '##', 'you', 'go', '##', 'i', 'go', '##']

    def test_boundary_between_every_sentence (self):
        assert corpus_to_words([['a'], ['b'], ['c']]) == [
            '##', 'a', '##', 'b', '##', 'c', '##']

    def test_single_sentence_still_bounded (self):
        assert corpus_to_words([['a', 'b']]) == ['##', 'a', 'b', '##']

    def test_empty_corpus (self):
        assert corpus_to_words([]) == ['##']


class TestGenFormVectors:
    def setup_method (self):
        self.ngrams = ['#ca', 'cat', 'at#']
        self.cue_index = {'#ca': 0, 'cat': 1, 'at#': 2}
        self.window = 2

    def test_shape (self):
        fv = gen_form_vectors(self.ngrams, self.cue_index, window=self.window)
        n_timesteps = len(self.ngrams) + self.window + 1
        assert fv.shape == (n_timesteps, len(self.cue_index))

    def test_t0_is_zeros (self):
        fv = gen_form_vectors(self.ngrams, self.cue_index, window=self.window)
        np.testing.assert_array_equal(fv[0], 0.0)

    def test_last_timestep_is_zeros (self):
        fv = gen_form_vectors(self.ngrams, self.cue_index, window=self.window)
        np.testing.assert_array_equal(fv[-1], 0.0)

    def test_t1 (self):
        fv = gen_form_vectors(self.ngrams, self.cue_index, window=self.window)
        np.testing.assert_array_almost_equal(fv[1], [1.0, 0.0, 0.0])

    def test_t2 (self):
        fv = gen_form_vectors(self.ngrams, self.cue_index, window=self.window)
        # '#ca' decayed to 0.5, 'cat' at 1.0
        np.testing.assert_array_almost_equal(fv[2], [0.5, 1.0, 0.0])

    def test_window4_linear_decay (self):
        """Linear decay over a four-item window."""
        words = ['dogs', 'are', 'always', 'best', 'friends']
        fv = gen_form_vectors(words, {w: i for i, w in enumerate(words)},
                              window=4)
        np.testing.assert_array_almost_equal(fv, [
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [1.00, 0.00, 0.00, 0.00, 0.00],
            [0.75, 1.00, 0.00, 0.00, 0.00],
            [0.50, 0.75, 1.00, 0.00, 0.00],
            [0.25, 0.50, 0.75, 1.00, 0.00],
            [0.00, 0.25, 0.50, 0.75, 1.00],
            [0.00, 0.00, 0.25, 0.50, 0.75],
            [0.00, 0.00, 0.00, 0.25, 0.50],
            [0.00, 0.00, 0.00, 0.00, 0.25],
            [0.00, 0.00, 0.00, 0.00, 0.00],
        ])


class TestIterFormVectors:
    def test_matches_gen_form_vectors (self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            length = int(rng.integers(0, 12))
            n_cues = int(rng.integers(1, 5))
            window = int(rng.integers(1, 6))
            alphabet = 'abcde'[:n_cues]
            seq = [alphabet[int(rng.integers(0, n_cues))] for _ in range(length)]
            cue_index = {c: i for i, c in enumerate(alphabet)}
            streamed = np.array(list(iter_form_vectors(seq, cue_index,
                                                       window=window)))
            np.testing.assert_array_equal(
                streamed, gen_form_vectors(seq, cue_index, window=window))

    def test_most_recent_occurrence_wins (self):
        # 'a' occurs at positions 0 and 2; at t=3 both are inside a window of 4
        fv = np.array(list(iter_form_vectors(['a', 'b', 'a', 'c'],
                                             {'a': 0, 'b': 1, 'c': 2}, window=4)))
        assert fv[3, 0] == 1.0        # the recent one, not 0.5 and not a sum

    def test_memory_is_independent_of_corpus_length (self):
        """The generator must not materialise the dense array."""
        tokens = corpus_to_words([['a', 'b', 'c']] * 400)
        cue_index = {c: i for i, c in enumerate(dict.fromkeys(tokens))}
        it = iter_form_vectors(tokens, cue_index, window=4)
        first = next(it)
        assert first.shape == (len(cue_index),)
        assert sum(1 for _ in it) == len(tokens) + 4


class TestCorpusWordTimesteps:
    def setup_method (self):
        self.sentences = [['you', 'go'], ['i', 'go']]

    def test_trigram_hits_last_ngram_of_word (self):
        ngrams = corpus_to_ngrams(self.sentences, gram=3)
        timesteps = _corpus_word_timesteps(self.sentences, 'trigram', 3)
        assert [ngrams[t - 1] for t in timesteps] == ['ou#', 'go#', '#i#', 'go#']

    def test_word_mode_hits_the_word (self):
        tokens = corpus_to_words(self.sentences)
        timesteps = _corpus_word_timesteps(self.sentences, 'word', 3)
        words = [w for sent in self.sentences for w in sent]
        assert [tokens[t - 1] for t in timesteps] == words

    def test_one_timestep_per_word (self):
        n_words = sum(len(s) for s in self.sentences)
        assert len(_corpus_word_timesteps(self.sentences, 'trigram', 3)) == n_words
        assert len(_corpus_word_timesteps(self.sentences, 'word', 3)) == n_words

    def test_timesteps_are_increasing (self):
        for mode in ('trigram', 'word'):
            ts = _corpus_word_timesteps(self.sentences, mode, 3)
            assert ts == sorted(ts) and len(set(ts)) == len(ts)


class TestPredictiveLearning:
    def setup_method (self):
        self.sentences = [
            ['the', 'cat', 'sat'],
            ['the', 'dog', 'ran'],
        ]
        self.semantic_dim = 10

    def test_output_keys (self):
        result = predictive_learning(self.sentences,
                                      semantic_dim=self.semantic_dim, randseed=42)
        assert 'F' in result and 'W' in result

    def test_F_dims (self):
        result = predictive_learning(self.sentences,
                                      semantic_dim=self.semantic_dim, randseed=42)
        assert result['F'].dims == ('cues', 'semantics')

    def test_W_dims (self):
        result = predictive_learning(self.sentences,
                                      semantic_dim=self.semantic_dim, randseed=42)
        assert result['W'].dims == ('semantics_from', 'semantics_to')

    def test_W_shape (self):
        result = predictive_learning(self.sentences,
                                      semantic_dim=self.semantic_dim, randseed=42)
        assert result['W'].shape == (self.semantic_dim, self.semantic_dim)

    def test_F_cues_dimension (self):
        result = predictive_learning(self.sentences,
                                      semantic_dim=self.semantic_dim,
                                      randseed=42, gram=3)
        n_cues = len(set(corpus_to_ngrams(self.sentences, gram=3)))
        assert result['F'].shape == (n_cues, self.semantic_dim)

    def test_boundary_cues_are_learned (self):
        result = predictive_learning(self.sentences,
                                      semantic_dim=self.semantic_dim, randseed=42)
        assert any('##' in cue for cue in result['F'].cues.values)

    def test_reproducibility (self):
        kw = dict(semantic_dim=self.semantic_dim, randseed=42)
        r1 = predictive_learning(self.sentences, **kw)
        r2 = predictive_learning(self.sentences, **kw)
        np.testing.assert_array_equal(r1['F'].values, r2['F'].values)
        np.testing.assert_array_equal(r1['W'].values, r2['W'].values)

    def test_different_seeds_differ (self):
        r1 = predictive_learning(self.sentences,
                                  semantic_dim=self.semantic_dim, randseed=42)
        r2 = predictive_learning(self.sentences,
                                  semantic_dim=self.semantic_dim, randseed=99)
        assert not np.array_equal(r1['F'].values, r2['F'].values)

    def test_weights_change_after_learning (self):
        result = predictive_learning(self.sentences,
                                      semantic_dim=self.semantic_dim, randseed=42)
        n_cues = len(set(corpus_to_ngrams(self.sentences, gram=3)))
        rng = np.random.default_rng(42)
        F_before = rng.normal(scale=0.01, size=(n_cues, self.semantic_dim))
        assert not np.array_equal(result['F'].values, F_before)

    def test_multiple_epochs (self):
        r1 = predictive_learning(self.sentences, semantic_dim=self.semantic_dim,
                                  epochs=1, randseed=42)
        r2 = predictive_learning(self.sentences, semantic_dim=self.semantic_dim,
                                  epochs=5, randseed=42)
        assert not np.array_equal(r1['F'].values, r2['F'].values)

    def test_bigrams (self):
        result = predictive_learning(self.sentences,
                                      semantic_dim=self.semantic_dim,
                                      gram=2, randseed=42)
        n_cues = len(set(corpus_to_ngrams(self.sentences, gram=2)))
        assert result['F'].shape == (n_cues, self.semantic_dim)

    def test_unknown_mode_raises (self):
        with pytest.raises(ValueError):
            predictive_learning(self.sentences, mode='sentence')

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_cpu_matches_numpy (self):
        kw = dict(semantic_dim=self.semantic_dim, randseed=42)
        r_np = predictive_learning(self.sentences, backend='numpy', **kw)
        r_th = predictive_learning(self.sentences, backend='torch',
                                    device='cpu', **kw)
        np.testing.assert_allclose(r_np['F'].values, r_th['F'].values, atol=1e-6)
        np.testing.assert_allclose(r_np['W'].values, r_th['W'].values, atol=1e-6)


class TestPredictiveLearningWordMode:
    def setup_method (self):
        self.sentences = [
            ['the', 'cat', 'sat'],
            ['the', 'dog', 'ran'],
        ]
        # '##' is an ordinary token and joins the vocabulary
        self.vocabulary = ['##', 'the', 'cat', 'sat', 'dog', 'ran']
        self.semantic_dim = 10

    def test_F_dims (self):
        result = predictive_learning(self.sentences,
                                      semantic_dim=self.semantic_dim,
                                      randseed=42, mode='word')
        assert result['F'].dims == ('word', 'semantics')

    def test_F_shape (self):
        result = predictive_learning(self.sentences,
                                      semantic_dim=self.semantic_dim,
                                      randseed=42, mode='word')
        assert result['F'].shape == (len(self.vocabulary), self.semantic_dim)

    def test_F_word_coords (self):
        result = predictive_learning(self.sentences,
                                      semantic_dim=self.semantic_dim,
                                      randseed=42, mode='word')
        assert list(result['F'].word.values) == self.vocabulary

    def test_boundary_token_is_learned (self):
        result = predictive_learning(self.sentences,
                                      semantic_dim=self.semantic_dim,
                                      randseed=42, mode='word')
        assert '##' in list(result['F'].word.values)

    def test_reproducibility (self):
        kw = dict(semantic_dim=self.semantic_dim, randseed=42, mode='word')
        np.testing.assert_array_equal(
            predictive_learning(self.sentences, **kw)['F'].values,
            predictive_learning(self.sentences, **kw)['F'].values)

    def test_weights_change (self):
        result = predictive_learning(self.sentences,
                                      semantic_dim=self.semantic_dim,
                                      randseed=42, mode='word')
        rng = np.random.default_rng(42)
        F_before = rng.normal(scale=0.01,
                              size=(len(self.vocabulary), self.semantic_dim))
        assert not np.array_equal(result['F'].values, F_before)

    def test_different_from_trigram_mode (self):
        kw = dict(semantic_dim=self.semantic_dim, randseed=42)
        r_word = predictive_learning(self.sentences, mode='word', **kw)
        r_tri = predictive_learning(self.sentences, mode='trigram', **kw)
        assert r_word['F'].shape != r_tri['F'].shape

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_cpu_matches_numpy (self):
        kw = dict(semantic_dim=self.semantic_dim, randseed=42, mode='word')
        r_np = predictive_learning(self.sentences, backend='numpy', **kw)
        r_th = predictive_learning(self.sentences, backend='torch',
                                    device='cpu', **kw)
        np.testing.assert_allclose(r_np['F'].values, r_th['F'].values, atol=1e-6)
        np.testing.assert_allclose(r_np['W'].values, r_th['W'].values, atol=1e-6)


class TestContinuousStream:
    """The corpus is one stream: the window spans sentence boundaries."""

    def setup_method (self):
        self.sentences = [['you', 'go'], ['i', 'go']]
        self.semantic_dim = 6

    def test_window_spans_the_boundary (self):
        tokens = corpus_to_words(self.sentences)
        cue_index = {c: i for i, c in enumerate(dict.fromkeys(tokens))}
        fv = gen_form_vectors(tokens, cue_index, window=4)
        # 'i' opens the second sentence at t=5; 'go' ended the first
        assert fv[5, cue_index['i']] == 1.0
        assert fv[5, cue_index['go']] > 0.0
        assert fv[5, cue_index['##']] > 0.0

    def test_second_sentence_has_a_prediction (self):
        """With no reset, the first word of sentence 2 is predicted from
        sentence 1 rather than from a zero vector."""
        model = predictive_learning(self.sentences,
                                     semantic_dim=self.semantic_dim,
                                     randseed=1, mode='word')
        F, W = model['F'].values, model['W'].values
        tokens = corpus_to_words(self.sentences)
        cue_index = {c: i for i, c in enumerate(model['F'].word.values)}
        fv = gen_form_vectors(tokens, cue_index, window=4)
        m_hat = (fv[4] @ F) @ W                   # prediction entering t=5
        assert np.linalg.norm(m_hat) > 0.0

    def test_sentence_order_matters (self):
        """Context crosses the boundary, so reordering sentences changes what
        is learned. Both corpora below have the same vocabulary in the same
        first-appearance order, so only the ordering of the stream differs."""
        kw = dict(semantic_dim=self.semantic_dim, randseed=1, mode='word')
        one = predictive_learning([['a'], ['b'], ['c'], ['b']], **kw)
        two = predictive_learning([['a'], ['b'], ['b'], ['c']], **kw)
        assert list(one['F'].word.values) == list(two['F'].word.values)
        assert not np.array_equal(one['F'].values, two['F'].values)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
class TestNumbaBackend:
    def setup_method (self):
        self.sentences = [
            ['the', 'cat', 'sat', 'on', 'the', 'mat'],
            ['the', 'dog', 'ran', 'away', 'fast'],
            ['a', 'cat', 'and', 'a', 'dog'],
        ]
        self.kw = dict(semantic_dim=8, randseed=42, epochs=3,
                       lr_f=0.05, lr_w=0.05)

    @pytest.mark.parametrize('mode', ['word', 'trigram'])
    def test_matches_numpy (self, mode):
        """Same arithmetic, different summation order, so allclose not equal."""
        pytest.importorskip('numba')
        r_np = predictive_learning(self.sentences, mode=mode,
                                    backend='numpy', **self.kw)
        r_nb = predictive_learning(self.sentences, mode=mode,
                                    backend='numba', **self.kw)
        np.testing.assert_allclose(r_np['F'].values, r_nb['F'].values, atol=1e-12)
        np.testing.assert_allclose(r_np['W'].values, r_nb['W'].values, atol=1e-12)

    def test_auto_prefers_numba (self):
        pytest.importorskip('numba')
        r_auto = predictive_learning(self.sentences, backend='auto', **self.kw)
        r_nb = predictive_learning(self.sentences, backend='numba', **self.kw)
        np.testing.assert_array_equal(r_auto['F'].values, r_nb['F'].values)

    def test_repeated_cue_still_most_recent (self):
        """The dedup inside the compiled loop must match iter_form_vectors."""
        pytest.importorskip('numba')
        sentences = [['a', 'b', 'a', 'c']]        # 'a' twice inside one window
        kw = dict(semantic_dim=6, randseed=1, epochs=2, mode='word')
        np.testing.assert_allclose(
            predictive_learning(sentences, backend='numpy', **kw)['F'].values,
            predictive_learning(sentences, backend='numba', **kw)['F'].values,
            atol=1e-12)

    def test_unknown_backend_raises (self):
        with pytest.raises(ValueError, match='Unknown backend'):
            predictive_learning(self.sentences, backend='bogus', **self.kw)


class TestPerSentenceMode:
    """continuous=False: each sentence on its own timeline, context reset."""

    def setup_method (self):
        self.sentences = [['the', 'cat', 'sat'], ['a', 'dog', 'ran', 'fast']]
        self.kw = dict(semantic_dim=8, randseed=1, epochs=3)

    def test_no_boundary_cue_in_vocabulary (self):
        word = predictive_learning(self.sentences, mode='word',
                                    continuous=False, **self.kw)
        assert '##' not in list(word['F'].word.values)
        tri = predictive_learning(self.sentences, mode='trigram',
                                   continuous=False, **self.kw)
        assert not any('##' in c for c in tri['F'].cues.values)

    def test_differs_from_continuous (self):
        a = predictive_learning(self.sentences, mode='word',
                                 continuous=True, **self.kw)
        b = predictive_learning(self.sentences, mode='word',
                                 continuous=False, **self.kw)
        assert a['F'].shape != b['F'].shape        # ## adds a row

    def test_sentence_order_still_matters (self):
        """What resets at a sentence boundary is the semantic context, not the
        weights: F and W carry across. So reversing the corpus order still
        changes the vectors, even though no context crosses a boundary."""
        kw = dict(semantic_dim=8, randseed=1, epochs=3, mode='word',
                  continuous=False)
        forward = predictive_learning([['a', 'b'], ['c', 'd']], **kw)
        reverse = predictive_learning([['c', 'd'], ['a', 'b']], **kw)
        assert set(str(w) for w in forward['F'].word.values) == \
               set(str(w) for w in reverse['F'].word.values)
        for w in ('a', 'b', 'c', 'd'):
            assert not np.allclose(forward['F'].sel(word=w).values,
                                   reverse['F'].sel(word=w).values)

    @pytest.mark.parametrize('mode', ['word', 'trigram'])
    def test_backends_agree (self, mode):
        pytest.importorskip('numba')
        kw = dict(mode=mode, continuous=False, **self.kw)
        np.testing.assert_allclose(
            predictive_learning(self.sentences, backend='numpy', **kw)['F'].values,
            predictive_learning(self.sentences, backend='numba', **kw)['F'].values,
            atol=1e-12)

    def test_consumers_run (self):
        model = predictive_learning(self.sentences, mode='trigram',
                                     continuous=False, **self.kw)
        wv = extract_word_vectors(self.sentences, model['F'], continuous=False)
        assert wv.dims == ('word', 'semantics')
        errors = prediction_errors(self.sentences, model['F'], model['W'],
                                    mode='trigram', continuous=False)
        assert [[w for w, _ in s] for s in errors] == self.sentences


class TestUpdateEvery:
    def setup_method (self):
        self.sentences = [['the', 'cat', 'sat', 'on', 'the', 'mat'],
                          ['a', 'dog', 'ran', 'away', 'very', 'fast']]
        self.kw = dict(semantic_dim=8, randseed=1, epochs=3)

    def test_default_is_every_timestep (self):
        a = predictive_learning(self.sentences, **self.kw)
        b = predictive_learning(self.sentences, update_every=1, **self.kw)
        np.testing.assert_array_equal(a['F'].values, b['F'].values)

    def test_changes_the_result (self):
        a = predictive_learning(self.sentences, update_every=1, **self.kw)
        b = predictive_learning(self.sentences, update_every=3, **self.kw)
        assert not np.array_equal(a['F'].values, b['F'].values)

    @pytest.mark.parametrize('continuous', [True, False])
    def test_backends_agree (self, continuous):
        pytest.importorskip('numba')
        kw = dict(mode='trigram', continuous=continuous, update_every=3,
                  **self.kw)
        np.testing.assert_allclose(
            predictive_learning(self.sentences, backend='numpy', **kw)['F'].values,
            predictive_learning(self.sentences, backend='numba', **kw)['F'].values,
            atol=1e-12)

    def test_prediction_errors_still_scores_every_word (self):
        """Words are scored at the last sampled step at or before their own."""
        model = predictive_learning(self.sentences, mode='trigram',
                                     update_every=3, **self.kw)
        errors = prediction_errors(self.sentences, model['F'], model['W'],
                                    mode='trigram', update_every=3)
        assert [[w for w, _ in s] for s in errors] == self.sentences

    def test_subsamples_the_timeline (self):
        """update_every=k visits ceil(n/k) of the n timesteps."""
        seq = corpus_to_words(self.sentences)
        cue_index = {c: i for i, c in enumerate(dict.fromkeys(seq))}
        n_all = sum(1 for _ in iter_form_vectors(seq, cue_index, window=4))
        from discriminative_lexicon_model.embedding import _iter_active_cues
        n_3 = sum(1 for _ in _iter_active_cues(seq, cue_index, window=4, step=3))
        assert n_3 == -(-n_all // 3)


class TestExtractWordVectors:
    def setup_method (self):
        self.sentences = [['you', 'go'], ['i', 'go']]
        self.semantic_dim = 6
        self.model = predictive_learning(self.sentences,
                                          semantic_dim=self.semantic_dim,
                                          randseed=1, mode='trigram')

    def test_one_vector_per_unique_word (self):
        wv = extract_word_vectors(self.sentences, self.model['F'])
        assert list(wv.word.values) == ['go', 'i', 'you']
        assert wv.shape == (3, self.semantic_dim)

    def test_dims (self):
        wv = extract_word_vectors(self.sentences, self.model['F'])
        assert wv.dims == ('word', 'semantics')

    def test_repeated_word_is_averaged (self):
        """'go' occurs twice, in different contexts, so it is a mean."""
        wv = extract_word_vectors(self.sentences, self.model['F'])
        assert np.isfinite(wv.sel(word='go').values).all()


class TestPredictionErrors:
    def setup_method (self):
        self.sentences = [
            ['the', 'cat', 'sat'],
            ['the', 'dog', 'ran'],
        ]
        self.semantic_dim = 8
        self.word = predictive_learning(self.sentences,
                                         semantic_dim=self.semantic_dim,
                                         randseed=42, mode='word')
        self.trigram = predictive_learning(self.sentences,
                                            semantic_dim=self.semantic_dim,
                                            randseed=42, mode='trigram')

    def test_one_error_per_word (self):
        result = prediction_errors(self.sentences, self.word['F'],
                                    self.word['W'], mode='word')
        assert [len(sent) for sent in result] == [len(s) for s in self.sentences]

    def test_words_match_input (self):
        result = prediction_errors(self.sentences, self.word['F'],
                                    self.word['W'], mode='word')
        assert [[w for w, _ in sent] for sent in result] == self.sentences

    def test_errors_are_non_negative (self):
        result = prediction_errors(self.sentences, self.word['F'],
                                    self.word['W'], mode='word')
        assert all(e >= 0.0 for sent in result for _, e in sent)

    def test_trigram_mode (self):
        result = prediction_errors(self.sentences, self.trigram['F'],
                                    self.trigram['W'], mode='trigram')
        assert [[w for w, _ in sent] for sent in result] == self.sentences

    def test_cosine_metric (self):
        result = prediction_errors(self.sentences, self.word['F'],
                                    self.word['W'], mode='word', metric='cosine')
        # cosine distance lies in [0, 2]
        assert all(0.0 <= e <= 2.0 for sent in result for _, e in sent)

    def test_cosine_of_a_zero_prediction_is_one (self):
        """Cosine is undefined against a zero vector, so it falls back to 1.0
        -- the orthogonal value, the midpoint of the [0, 2] range and not its
        maximum. Per-sentence mode resets s_prev to zeros, making the
        prediction entering the first word exactly zero."""
        model = predictive_learning(self.sentences, mode='word',
                                     semantic_dim=4, randseed=0,
                                     continuous=False)
        result = prediction_errors(self.sentences[:1], model['F'], model['W'],
                                    mode='word', metric='cosine',
                                    continuous=False)
        assert result[0][0][1] == 1.0

    def test_reproducibility (self):
        kw = dict(mode='word')
        r1 = prediction_errors(self.sentences, self.word['F'],
                                self.word['W'], **kw)
        r2 = prediction_errors(self.sentences, self.word['F'],
                                self.word['W'], **kw)
        assert r1 == r2

    def test_first_word_is_predicted_from_the_boundary (self):
        """t=1 is the '##' token, so the first word already has a non-zero
        prediction behind it -- unlike the old per-sentence reset."""
        result = prediction_errors(self.sentences, self.word['F'],
                                    self.word['W'], mode='word')
        F, W = self.word['F'].values, self.word['W'].values
        tokens = corpus_to_words(self.sentences)
        cue_index = {c: i for i, c in enumerate(self.word['F'].word.values)}
        fv = gen_form_vectors(tokens, cue_index, window=4)
        m_hat = (fv[1] @ F) @ W               # prediction entering t=2 ('the')
        np.testing.assert_allclose(result[0][0][1],
                                   np.linalg.norm(fv[2] @ F - m_hat))


class TestSaveLoadModel:
    def setup_method (self):
        self.sentences = [['you', 'go'], ['i', 'go']]

    @pytest.mark.parametrize('mode', ['word', 'trigram'])
    def test_roundtrip_values (self, tmp_path, mode):
        model = predictive_learning(self.sentences, semantic_dim=5,
                                     randseed=0, mode=mode)
        path = tmp_path / 'model.nc'
        save_model(model, str(path))
        loaded = load_model(str(path))
        np.testing.assert_allclose(model['F'].values, loaded['F'].values)
        np.testing.assert_allclose(model['W'].values, loaded['W'].values)

    @pytest.mark.parametrize('mode,form_dim', [('word', 'word'),
                                               ('trigram', 'cues')])
    def test_roundtrip_preserves_labels (self, tmp_path, mode, form_dim):
        """prediction_errors and extract_word_vectors look words up by F's
        coordinate labels, not by row number, so dims and labels must survive
        the round trip and not just the numbers."""
        model = predictive_learning(self.sentences, semantic_dim=5,
                                     randseed=0, mode=mode)
        path = tmp_path / 'model.nc'
        save_model(model, str(path))
        loaded = load_model(str(path))
        assert loaded['F'].dims == (form_dim, 'semantics')
        assert loaded['W'].dims == ('semantics_from', 'semantics_to')
        assert list(loaded['F'][form_dim].values) == list(model['F'][form_dim].values)
        assert list(loaded['F'].semantics.values) == list(model['F'].semantics.values)

    def test_loaded_model_still_scores (self, tmp_path):
        """A reloaded model must be usable, not merely equal."""
        model = predictive_learning(self.sentences, semantic_dim=5,
                                     randseed=0, mode='word')
        path = tmp_path / 'model.nc'
        save_model(model, str(path))
        loaded = load_model(str(path))
        assert (prediction_errors(self.sentences, loaded['F'], loaded['W'],
                                  mode='word')
                == prediction_errors(self.sentences, model['F'], model['W'],
                                     mode='word'))
