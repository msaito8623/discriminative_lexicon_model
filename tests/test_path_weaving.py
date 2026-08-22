import numpy as np
import pandas as pd
import xarray as xr

import pytest

import discriminative_lexicon_model.mapping as dm
import discriminative_lexicon_model.path_weaving as dw

# This dataframe, `dat`, comes from the page 45 (i.e., "Code 3.5") of Heimeier,
# Chuang, & Baayen (2026). It is only the first four rows of the entire dataset
# discussed in the book.
dat = pd.DataFrame({'Ortho'  : ['aap', 'aard', 'aars', 'aas'],
                      'Lexeme' : ['aap', 'aard', 'aars', 'aas'],
                      'Number' : ['singular', 'singular', 'singular', 'singular'],
                      'WordCat': ['noun', 'verb', 'noun', 'noun']})


def book_matrices (dim_size=3, seed=10):
    cmat = dm.gen_cmat(dat['Ortho'], gram=2)
    smat = dm.gen_smat_sim(dat, form='Ortho', include_form=False,
                           dim_size=dim_size, seed=seed)
    fmat = dm.gen_fmat(cmat=cmat, smat=smat)
    gmat = dm.gen_gmat(smat=smat, cmat=cmat)
    chat = dm.gen_chat(smat=smat, gmat=gmat)
    return cmat, smat, fmat, chat


class TestCueSequence:
    def test_order_is_kept (self):
        assert dw.cue_sequence('aap', gram=2) == ['#a', 'aa', 'ap', 'p#']

    def test_duplicates_are_kept (self):
        assert dw.cue_sequence('banana', gram=3).count('ana') == 2


class TestGenAmat:
    def test_cue_set_of_page_87 (self):
        cues = list(dm.gen_cmat(dat['Ortho'], gram=2).cues.values)
        cues_book = ['#a', 'aa', 'ap', 'p#', 'ar', 'rd', 'd#', 'rs', 'as', 's#']
        assert sorted(cues) == sorted(cues_book)

    def test_attested_transitions_only (self):
        amat = dw.gen_amat(dat['Ortho'], gram=2)
        assert bool(amat.sel(current='#a', next='aa'))
        assert not bool(amat.sel(current='#a', next='ap')) # Valid but not in the dataset.
        assert not bool(amat.sel(current='p#', next='#a'))
        assert not bool(amat.sel(current='aa', next='aa'))

    def test_is_a_strict_subset_of_vmat (self):
        amat = dw.gen_amat(dat['Ortho'], gram=2)
        cues = dm.to_cues(dat['Ortho'], gram=2)
        vmat = dm.gen_vmat(cues=cues).sel(current=cues)
        assert not (amat.values & ~vmat.values).any()
        assert amat.values.sum() < vmat.values.sum()


class TestGenYmats:
    def test_count_is_the_longest_word (self):
        cmat, _, _, _ = book_matrices()
        assert len(dw.gen_ymats(cmat)) == 5

    def test_y1_marks_the_first_cue (self):
        cmat, _, _, _ = book_matrices()
        y1 = dw.gen_ymats(cmat)[0]
        assert (y1.sel(cues='#a').values == 1).all()
        assert y1.values.sum() == 4

    def test_y3_of_page_87 (self):
        cmat, _, _, _ = book_matrices()
        y3 = dw.gen_ymats(cmat)[2]
        for word, cue in [('aap','ap'), ('aard','ar'), ('aars','ar'), ('aas','as')]:
            assert float(y3.sel(word=word, cues=cue)) == 1.0
        assert float(np.array(y3).sum()) == 4.0

    def test_y5_of_page_88 (self):
        cmat, _, _, _ = book_matrices()
        y5 = dw.gen_ymats(cmat)[4]
        assert float(y5.sel(word='aard', cues='d#')) == 1.0
        assert float(y5.sel(word='aars', cues='s#')) == 1.0
        assert float(y5.sel(word='aap').values.sum()) == 0.0
        assert float(y5.sel(word='aas').values.sum()) == 0.0
        assert float(y5.values.sum()) == 2.0

    def test_rows_hold_one_cue_at_most (self):
        cmat, _, _, _ = book_matrices()
        yms = dw.gen_ymats(cmat)
        for y in yms:
            assert set(y.values.sum(axis=1)) <= {0.0, 1.0}


class TestGenMmats:
    def test_one_mapping_per_position (self):
        cmat, _, _, _ = book_matrices()
        ymats = dw.gen_ymats(cmat)
        assert len(dw.gen_mmats(cmat, ymats)) == len(ymats)

    def test_shared_core_matches_the_direct_estimate (self):
        cmat, _, _, _ = book_matrices()
        ymats = dw.gen_ymats(cmat)
        mmats = dw.gen_mmats(cmat, ymats)
        c = cmat.values
        for ymat, mmat in zip(ymats, mmats):
            m_direct = np.linalg.pinv(c.T @ c) @ c.T @ ymat.values
            assert np.allclose(mmat.values, m_direct)


class TestGenYhats:
    def test_recovers_ymats_when_chat_is_exact (self):
        # cols > rows: Should be perfect in prediction.
        cmat, _, _, chat = book_matrices(dim_size=10)
        assert np.allclose(chat.values, cmat.values)
        ymats = dw.gen_ymats(cmat)
        mmats = dw.gen_mmats(cmat, ymats)
        yhats = dw.gen_yhats(chat, mmats)
        for ymat, yhat in zip(ymats, yhats):
            assert np.allclose(yhat.values, ymat.values, atol=1e-8)

    def test_argmax_is_the_true_positional_cue (self):
        # cols < rows: Should not be perfect in prediction.
        cmat, _, _, chat = book_matrices(dim_size=3)
        ymats = dw.gen_ymats(cmat)
        mmats = dw.gen_mmats(cmat, ymats)
        yhats = dw.gen_yhats(chat, mmats)
        for word in cmat.word.values:
            seq = dw.cue_sequence(word, gram=2)
            for pos, cue in enumerate(seq):
                row = yhats[pos].sel(word=word)
                maxpos = int(np.argmax(row.values))
                assert str(row.cues.values[maxpos]) == cue


class TestFindPathsBox82:
    def setup_method (self):
        cmat, smat, fmat, _ = book_matrices()
        self.cmat = cmat
        self.smat = smat
        self.fmat = fmat
        words = cmat.word.values.tolist()
        cues = cmat.cues.values.tolist()
        self.vmat = dm.gen_vmat(cues=cues)
        self.amat = dw.gen_amat(words, gram=2)
        supported = ['#a', 'aa', 'ap', 'as', 'ar', 'p#', 's#']
        vals = [ [ 1.0 if (c in supported) else 0.0 for c in cues ] ]
        chat = xr.DataArray(vals, dims=('word', 'cues'), coords={'word': ['aap'], 'cues': cues})
        self.chat = chat
        self.yhats = [ chat.copy() for _ in range(5) ]

    def forms (self, paths):
        x = [dm._concat_selected(p, overlap=True).strip('#') for p in paths]
        x = sorted(x)
        return x

    def test_the_four_candidates_of_the_book (self):
        paths = dw.find_paths(self.chat, self.yhats, self.vmat, threshold=0.5)
        forms = self.forms(paths)
        forms_book = ['aap', 'aas', 'ap', 'as']
        assert set(forms_book) <= set(forms)

    def test_aar_and_ar_are_not_produced (self):
        paths = dw.find_paths(self.chat, self.yhats, self.vmat, threshold=0.5)
        forms = self.forms(paths)
        assert 'aar' not in forms
        assert 'ar' not in forms

    def test_paths_start_initial_and_end_final (self):
        paths = dw.find_paths(self.chat, self.yhats, self.vmat, threshold=0.5)
        assert len(paths) > 0
        for path in paths:
            assert path[0].startswith('#')
            assert path[-1].endswith('#')

    @pytest.mark.parametrize('use_amat', [True, False])
    def test_no_form_runs_through_a_word_boundary (self, use_amat):
        amat = self.amat if use_amat else None
        paths = dw.find_paths(self.chat, self.yhats, self.vmat, amat=amat,
                              threshold=0.5)
        forms = self.forms(paths)
        is_boundary_free = [ ('#' not in f) for f in forms]
        assert all(is_boundary_free)

    def test_synthesis_by_analysis_picks_the_target (self):
        paths = dw.find_paths(self.chat, self.yhats, self.vmat, threshold=0.5)
        gold = self.smat.sel(word='aap')
        winner, corrs = dw.select_path(paths, gold, self.cmat, self.fmat)
        assert winner == 'aap'
        assert 'ap' in corrs
        assert 'as' in corrs

    def test_amat_prunes_the_self_loop (self):
        paths = dw.find_paths(self.chat, self.yhats, self.vmat, amat=self.amat,
                              threshold=0.5)
        forms = self.forms(paths)
        assert forms == ['aap', 'aas']


class TestFindPathsWord:
    def setup_method (self):
        self.cmat, self.smat, self.fmat, self.chat = book_matrices()
        self.vmat = dm.gen_vmat(cues=list(self.cmat.cues.values))
        self.amat = dw.gen_amat(dat['Ortho'], gram=2)
        self.yhats = dw.gen_yhats(self.chat,
                                  dw.gen_mmats(self.cmat, dw.gen_ymats(self.cmat)))

    @pytest.mark.parametrize('word', ['aap', 'aard', 'aars', 'aas'])
    def test_gold_path_is_found (self, word):
        paths = dw.find_paths(self.chat, self.yhats, self.vmat, word=word,
                              amat=self.amat, threshold=0.1)
        paths = [list(p) for p in paths] # Tuple to list
        assert dw.cue_sequence(word, gram=2) in paths

    def test_word_may_be_an_index (self):
        by_label = dw.find_paths(self.chat, self.yhats, self.vmat, word='aars',
                                 amat=self.amat)
        by_index = dw.find_paths(self.chat, self.yhats, self.vmat, word=2,
                                 amat=self.amat)
        assert by_label == by_index

    def test_word_is_required_for_several_rows (self):
        with pytest.raises(ValueError):
            dw.find_paths(self.chat, self.yhats, self.vmat, amat=self.amat)

    def test_word_may_be_dropped_for_a_single_row (self):
        one = self.chat.sel(word=['aap'])
        yhats = [y.sel(word=['aap']) for y in self.yhats]
        a = dw.find_paths(one, yhats, self.vmat, amat=self.amat)
        b = dw.find_paths(self.chat, self.yhats, self.vmat, word='aap', amat=self.amat)
        assert a == b

    def test_high_threshold_finds_nothing (self):
        paths = dw.find_paths(self.chat, self.yhats, self.vmat, word='aap',
                              amat=self.amat, threshold=1.5)
        assert paths == []


class TestSelectPath:
    def setup_method (self):
        self.cmat, self.smat, self.fmat, self.chat = book_matrices()

    def test_gold_form_wins (self):
        words = self.cmat.word.values
        paths = [dw.cue_sequence(w, gram=2) for w in words]
        gold = self.smat.sel(word='aard')
        winner, corrs = dw.select_path(paths, gold, self.cmat, self.fmat)
        assert winner == 'aard'
        assert len(corrs) == 4

    def test_no_paths_gives_an_empty_form (self):
        paths = []
        gold = self.smat.sel(word='aap')
        form_corrs = dw.select_path(paths, gold, self.cmat, self.fmat)
        assert form_corrs == ('', {})

    def test_nan_candidate_never_wins (self):
        fmat = self.fmat.copy()
        for cue in dw.cue_sequence('aas', gram=2):
            fmat.loc[{fmat.dims[0]: cue}] = 1.0
        paths = [dw.cue_sequence(w, gram=2) for w in ['aas', 'aap']]
        with np.errstate(invalid='ignore'): # Suppress a warning.
            gold = self.smat.sel(word='aap')
            winner, corrs = dw.select_path(paths, gold, self.cmat, fmat)
        assert np.isnan(corrs['aas'])
        assert winner == 'aap'

    def test_all_nan_gives_an_empty_form (self):
        gold = xr.zeros_like(self.smat.sel(word='aap')) + 1.0
        words = self.smat.word.values.tolist()
        paths = [dw.cue_sequence(w, gram=2) for w in words]
        with np.errstate(invalid='ignore'):
            winner, corrs = dw.select_path(paths, gold, self.cmat, self.fmat)
        assert winner == ''
        corrs = [np.isnan(i) for i in corrs.values()]
        assert all(corrs)


class TestWeave:
    def setup_method (self):
        self.cmat, self.smat, self.fmat, self.chat = book_matrices()
        words = self.cmat.word.values.tolist()
        cues = self.cmat.cues.values.tolist()
        self.vmat = dm.gen_vmat(cues=cues)
        self.amat = dw.gen_amat(words, gram=2)
        self.ymats = dw.gen_ymats(self.cmat)
        self.mmats = dw.gen_mmats(self.cmat, self.ymats)
        self.yhats = dw.gen_yhats(self.chat, self.mmats)

    @pytest.mark.parametrize('word', ['aap', 'aard', 'aars', 'aas'])
    def test_produces_the_targeted_form (self, word):
        gold = self.smat.sel(word=word)
        form, corrs = dw.weave(gold, self.cmat, self.fmat, self.chat,
                               self.yhats, self.vmat, word=word,
                               amat=self.amat)
        assert form == word

    def test_agrees_with_find_paths_and_select_path (self):
        paths = dw.find_paths(self.chat, self.yhats, self.vmat, word='aas', amat=self.amat)
        gold = self.smat.sel(word='aas')
        expected = dw.select_path(paths, gold, self.cmat, self.fmat)
        form_corrs = dw.weave(gold, self.cmat, self.fmat, self.chat,
                              self.yhats, self.vmat, word='aas',
                              amat=self.amat)
        assert form_corrs == expected
