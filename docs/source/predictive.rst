===================
Predictive learning
===================

``discriminative_lexicon_model.embedding`` builds up embeddings (semantic
representations) from running text. Unlike the estimation of
:math:`\mathbf{F}`, no target semantics :math:`\mathbf{S}` needs to be
supplied. The model makes its own prediction about the next semantic state,
which gets compared against what the model really understands in the next step.
The errors between the predicted and actual semantic states serve as the
learning signal.

Note that this algorithm is different from the one in :ref:`Incremental
learning`, which updates a weight matrix :math:`\mathbf{F}` incrementally
towards a gold-standard semantic matrix supplied in advance.



-----------
Basic usage
-----------
The main method of this predictive learning algorithm is implemented as
``predictive_learning``. The method receives the corpus (input text) and
returns the two weight matrices as a dictionary, i.e. ``'F'`` and ``'W'``. The
first weight matrix :math:`\mathbf{F}` is the same :math:`\mathbf{F}` as in DLM
in general, which maps forms onto meanings. The second weight matrix
:math:`\mathbf{W}` maps the current semantics onto the next predicted
semantics. See below the "Learning Mechanism" section for more details.

.. code-block:: python

    >>> import discriminative_lexicon_model.embedding as dlm
    >>> corpus = [['you', 'go'], ['i', 'go']]
    >>> model = dlm.predictive_learning(corpus, mode='word', gram=3, window=4,
    ...                                 continuous=True, update_every=1,
    ...                                 semantic_dim=10, epochs=5, randseed=0)

    >>> model['F'].shape
    (4, 10)
    >>> model['F'].dims
    ('word', 'semantics')
    >>> model['F'].coords['word'].values.tolist()
    ['##', 'you', 'go', 'i']
    >>> model['F'].coords['semantics'].values.tolist()[:3]
    ['S000', 'S001', 'S002']

    >>> model['W'].shape
    (10, 10)
    >>> model['W'].dims
    ('semantics_from', 'semantics_to')
    >>> model['W'].coords['semantics_from'].values.tolist()[:3]
    ['S000', 'S001', 'S002']
    >>> model['W'].coords['semantics_to'].values.tolist()[:3]
    ['S000', 'S001', 'S002']

    >>> dlm.save_model(model, 'model.nc')
    >>> model = dlm.load_model('model.nc')

It is possible and therefore remains an empirical question how often the
listener/reader makes a prediction about the next (semantic) state. Especially
in the ngram mode, making a prediction and updating the matrices for every
ngram might be too frequent. ``update_every=k`` controls how often the weights
get updated: weights get updated at every :math:`k`-th timestep.

Models' prediction errors can be calculated by ``prediction_errors``, which
does not update the weights and only returns prediction errors, i.e., one
``(word, error)`` pair for each token. Prediction errors are calculated either
as Euclidean distances (``metric='euclidean'``) or as cosine distances
(``metric='cosine'``). The parameters, ``mode``, ``gram``, ``window``,
``continuous``, and ``update_every`` must be matched between
``predictive_learning`` and ``prediction_errors``.

.. code-block:: python

    >>> dlm.prediction_errors(corpus, model['F'], model['W'], mode='word',
    ...                       gram=3, window=4, continuous=True, update_every=1,
    ...                       metric='euclidean')
    [[('you', np.float64(0.026717371130704656)),
      ('go', np.float64(0.027328136016373687))],
     [('i', np.float64(0.018602667769771854)),
      ('go', np.float64(0.022112023223215153))]]




------------------
Learning mechanism
------------------

In this algorithm, two matrices are learned, which are :math:`\mathbf{F}`
mapping a form vector to a semantic vector and :math:`\mathbf{W}` predicting
the next semantic vector from the current one.

At timestep :math:`t`, the model receives the form vector :math:`\mathbf{c}_{t}`
and produces its semantic understanding of the input, namely
:math:`\hat{\mathbf{s}}_{t}`, based on what it has learned so far, regarding
the relationships between form and meaning, namely :math:`\mathbf{F}_{t}`:

.. math::

    \hat{\mathbf{s}}_{t} = \mathbf{c}_{t} \mathbf{F}_{t}

Based on the understood meaning :math:`\hat{\mathbf{s}}_{t}`, the model
predicts what it will have understood by the next time step, i.e.
:math:`\hat{\mathbf{m}}_{t}`. This is achieved by the weight matrix
:math:`\mathbf{W}_{t}` mapping the current understood meaning to the next predicted
meaning:

.. math::

    \hat{\mathbf{m}}_{t+1} = \hat{\mathbf{s}}_{t} \mathbf{W}_{t}

Therefore, the model's prediction about the semantic state made in the previous
step about the current step is represented as below:

.. math::

    \hat{\mathbf{m}}_{t} = \hat{\mathbf{s}}_{t-1} \mathbf{W}_{t-1}

The errors between the semantic state predicted in the last step and the actual
semantic state that has been understood based on the current form input serve
as error signals and are represented as below:

.. math::

    \mathbf{e}_{t} = \hat{\mathbf{s}}_{t} - \hat{\mathbf{m}}_{t}

This error vector will serve to update the directly adjacent weight matrix
:math:`\mathbf{W}` later. In addition, this error vector can be back-propagated
by one layer as below:

.. math::

    \boldsymbol{\delta}_{t} = \mathbf{e}_{t} \mathbf{W}_{t}^{\top}

where :math:`\boldsymbol{\delta}` represents the errors that can be
attributable to the output layer of the first mapping, conceptually serving as
the "gold-standard" vector for the layer.

With one error vector for the output of the first mapping (i.e.,
:math:`\boldsymbol{\delta}_{t}`) and another error vector for the output of the
second mapping (i.e., :math:`\mathbf{e}_{t}`), the two weight matrices
:math:`\mathbf{F}_{t}` and :math:`\mathbf{W}_{t}` can be updated in the Widrow-Hoff
way:

.. math::

    \mathbf{W}_{t+1} &= \mathbf{W}_{t} + \eta_w \, \hat{\mathbf{s}}_{t-1}^\top \mathbf{e}_{t} \\
    \mathbf{F}_{t+1} &= \mathbf{W}_{F} + \eta_f \, \mathbf{c}_{t-1}^\top \boldsymbol{\delta}_{t}

:math:`\mathbf{F}` has tokens (e.g., words) as rows and semantic dimensions as
columns. Therefore, after sufficient training, the row vectors of
:math:`\mathbf{F}` represent embeddings of the tokens.


------------
Form vectors
------------
A input form vector indicates which tokens are activated to what extent in a
sliding window. Activation is assumed to decrease linearly for the current
implementation. The window size is a hyper-parameter. For example, for the
window size of 4, the newest token receives the activation of :math:`1.0`, the
second newest (previous) token receives :math:`0.75`, the third newest (second
oldest) :math:`0.50`, and the oldest `0.25`. 

To generalize, let a token's *age* be :math:`a`, which represents how many
steps back from the newest token, with the newest item being :math:`a = 0`.
Then activation values are represented by:

.. math::

    1 - \frac{a}{\text{window}}

If the same tokens occur multiple times inside one window, the activaiton value
for the most recent one will be taken. In other words, activation values are
not summed.

Form vectors utilized for this algorithm can be generated by
``gen_form_vectors``, where ``seq`` represents input text (e.g., a sentence)
and ``idx`` represents the indice, namely the column positions, of the tokens:

.. code-block:: python

    >>> import discriminative_lexicon_model as dlm
    >>> seq = ['a', 'b', 'c']
    >>> idx = {'a': 0, 'b': 1, 'c': 2}
    >>> dlm.gen_form_vectors(seq, idx, window=2)

    array([[0. , 0. , 0. ],
           [1. , 0. , 0. ],
           [0.5, 1. , 0. ],
           [0. , 0.5, 1. ],
           [0. , 0. , 0.5],
           [0. , 0. , 0. ]])




-------------------
Sentence boundaries
-------------------
The corpus is supplied either as a list of sentences, e.g.:

.. code-block:: python

   >>> print([['you', 'go'], ['i', 'go']])
   [['you', 'go'], ['i', 'go']]

or as a list of words/tokens, concatenating sentences:

.. code-block:: python

    >>> dlm.corpus_to_words([['you', 'go'], ['i', 'go']])
    ['##', 'you', 'go', '##', 'i', 'go', '##']

When the corpus is supplied as separate sentences, these sentences are treated
individually, meaning: the order of these sentences will be discarded.

When the corpus is supplied as a single stream of text, in contrast, sentence
boundaries are marked by the doubled boundary character, ``'##'``, and the
order of the sentences will naturally be taken into account. The sentence
boundary token will be treated in the same way as other real tokens such as
words.

A list of sentences can be collapsed into a single stream of text by
``corpus_to_words`` for words and ``corpus_to_ngrams`` for ngrams:

.. code-block:: python

    >>> dlm.corpus_to_words([['you', 'go'], ['i', 'go']])
    ['##', 'you', 'go', '##', 'i', 'go', '##']

    >>> dlm.corpus_to_ngrams([['you', 'go'], ['i', 'go']], gram=3)
    ['##y', '#yo', 'you', 'ou#', 'u#g', '#go', 'go#', 'o##',
     '##i', '#i#', 'i#g', '#go', 'go#', 'o##']

In the ngram mode, words are chunked into ngrams and word boundaries are marked
by the single boundary character ``'#'``. Sentence boundaries are marked by the
double boundary character ``'##'``. 

By concatenating sentences into a single stream of text, the corpus is treated
as connected discourse. By default, ``predictive_learning`` assumes the
continuous input (i.e., ``continuous=True``. When sentences are unordered or
unrelated, ``continuous=False`` will treat each sentence separately.

