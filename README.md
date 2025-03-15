# RASP-for-COGS

To support Kim and Linzen 2020's original COGS format ( https://aclanthology.org/2020.emnlp-main.731/ ) (vs the simplified and semantically equivalent ReCOGS from Wu et al 2023, which we built a model for earlier in RASP described at https://github.com/willy-b/RASP-for-ReCOGS ).

To try this out, run `python cogs_examples_in_rasp.py`!

Evaluation on test and evaluation on gen sets are disabled for now as this is still development phase but the provided script `cogs_examples_in_rasp.py` can be used to evaluate on training or dev set data.

Context:
It is simpler and runs faster to just train a Transformer on examples!

This is an academic exercise, writing a neural network compatible program by hand in the Restricted Access Sequence Processing (compilable to Transformer) language (Weiss et al 2021, https://arxiv.org/abs/2106.06981 )
to prove a Transformer can perform a particular type of solution (building a systematic and compositional solution without a tree-structured/hierarchical representation (flat attention-head based pattern matching) that can handle recursive grammar by unrolling it in the decoder loop).

**We want to show a Transformer can solve the structural generalization splits of the "COGS: A Compositional Generalization Challenge Based on Semantic Interpretation"**
(Kim and Linzen 2020, https://aclanthology.org/2020.emnlp-main.731 ) task in the original COGS format - **Transformers trained from scratch on the training split have been reported in the literature to do as poorly as 0% accuracy on the structural generalizations by multiple authors.**

(We referenced the convenient summary of Kim, Linzen's COGS task grammar and vocab made by IBM's CPG project (their utilities, not their CPG project itself) at https://github.com/IBM/cpg/blob/c3626b4e03bfc681be2c2a5b23da0b48abe6f570/src/model/cogs_data.py#L523 during development, restricting ourselves to the vocabulary available in the training data.)

See also our finished RASP solution to the semantically equivalent ReCOGS task (Wu et al 2023, "ReCOGS: How Incidental Details of a Logical Form Overshadow an Evaluation of Semantic Interpretation", https://arxiv.org/abs/2303.13716) at https://github.com/willy-b/learning-rasp/blob/main/word-level-pos-tokens-recogs-style-decoder-loop.rasp 

with draft paper for the ReCOGS solution at https://raw.githubusercontent.com/willy-b/RASP-for-ReCOGS/main/rasp-for-recogs_pos-wbruns-2024-draft.pdf for context on this work.

Note this work is focused on structural generalization, not lexical generalization.