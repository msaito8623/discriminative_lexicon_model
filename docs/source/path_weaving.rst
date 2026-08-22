============
Path weaving
============

In the production process in the framework of the Discriminative Lexicon Model,
a semantic vector is mapped onto a predicted form vector,
:math:`\hat{\mathbf{c}} = \mathbf{s}\mathbf{G}`. The predicted form vector only
indicates which cue (e.g., bigram) is activated / supported by the meaning of
the word to what extent. However, it is not clearly indicated which cues to be
selected and in which order they should be concatenated. The *path-weaving*
algorithm, explained in detail in Heitmeier, Chuang, & Baayen (2026), achieves
this task, namely to select which cues to be considered and how to concatenate
them.

The path-weaving algorithm consists of four steps.

1. *positional learning* 

   In addition to a :math:`\mathbf{C}` matrix, which indicates which cues are a part of which words, binary matrices :math:`\mathbf{Y}_n` are also required. The positional matrices, :math:`\mathbf{Y}_n`, indicate which cue is present in which word at the position :math:`n`.  From :math:`\mathbf{C}` and :math:`\mathbf{Y}_n`, positional mapping matrices :math:`\mathbf{M}_n` are estimated. There is one mapping matrix, :math:`\mathbf{M}_n`, per position. The estimated positional matrices, :math:`\mathbf{M}_n`, are then used to map :math:`\hat{\mathbf{C}}` onto :math:`\hat{\mathbf{Y}}_n`, i.e., the speaker's estimation about which cue is likely to what extent at the given position for the word. Note that :math:`\hat{\mathbf{Y}}_n` is estimated with :math:`\hat{\mathbf{C}}`, a predicted form matrix, which is originally mapped from :math:`\mathbf{S}`.

2. *thresholding*

   In the path-weaving algorithm, not every cue is considered. Defining a cue set that is to be considered is a part of this algorithm. This candidate cue set is determined by semantic support values, i.e. the values in the predicted form matrix :math:`\hat{\mathbf{C}}`, and positional support, i.e., the values in the predicted positional matrix :math:`\hat{\mathbf{Y}}_n`. Only the cues that exceed a certain threshold for both criteria will be admitted to the further process.


3. *path finding*

   The remaining cues are considered with respect to their possible concatenations. To prune non-word concatenations, the algorithm makes use of an adjacency matrix :math:`\mathbf{A}`, which encodes which cues can occur after which cues. This adjacency matrix is constructed from the training data. Therefore, only the transitions encountered in the training are considered to be valid. During the path-finding process, the algorithm starts from the possible word-initial cues and considers possible next cues for each of them. Since one cue can be followed by multiple different cues, these paths grow and diverge (and converge as well). Complete paths, namely those beginning with word-initial cues and ending with word-final cues, are considered for the next step.

4. *synthesis-by-analysis*

   The previous step, the path finding, does not pin down one path yet. It only enumerates possible strings of cues, namely possible word forms. After the enumeration of word forms, the synthesis-by-analysis determines which word form is ultimately the winner. To determine the winner, the algorithm maps every candidate word form back onto semantics, :math:`\hat{\mathbf{S}}_{cand} = \mathbf{C}_{cand}\mathbf{F}`. By doing so, each candidate word form has its own predicted semantic vector. These predicted semantic vectors are then compared against the correct, gold-standard semantic vector, from which this endeavour of speech production has started. The candidate word form gets selected as a winner, when it generates the predicted semantic vector that is the closest to the correct semantic vector.

.. code-block:: python

    >>> import pandas as pd
    >>> import discriminative_lexicon_model.mapping as dm
    >>> import discriminative_lexicon_model.path_weaving as dw

    >>> df = pd.DataFrame({'Ortho'  : ['aap', 'aard', 'aars', 'aas'],
    ...                    'Lexeme' : ['aap', 'aard', 'aars', 'aas'],
    ...                    'Number' : ['singular', 'singular', 'singular', 'singular'],
    ...                    'WordCat': ['noun', 'verb', 'noun', 'noun']})
    >>> cmat = dm.gen_cmat(df['Ortho'], gram=2)
    >>> smat = dm.gen_smat_sim(df, form='Ortho', include_form=False, dim_size=6, seed=1)
    >>> fmat = dm.gen_fmat(cmat=cmat, smat=smat)
    >>> gmat = dm.gen_gmat(smat=smat, cmat=cmat)
    >>> chat = dm.gen_chat(smat=smat, gmat=gmat)

    >>> ymats = dw.gen_ymats(cmat)
    >>> mmats = dw.gen_mmats(cmat, ymats)
    >>> yhats = dw.gen_yhats(chat, mmats)
    >>> vmat  = dm.gen_vmat(cues=list(cmat.cues.values))
    >>> amat  = dw.gen_amat(df['Ortho'], gram=2)

    >>> dw.find_paths(chat, yhats, vmat, word='aap', amat=amat, threshold=0.1)
    [('#a', 'aa', 'ap', 'p#')]

    >>> dw.weave(smat.sel(word='aap'), cmat, fmat, chat, yhats, vmat, word='aap', amat=amat)
    ('aap', {'aap': 1.0})

