# RASP-for-COGS

To support Kim and Linzen 2020's original COGS format ( https://aclanthology.org/2020.emnlp-main.731/ ) (vs the simplified and semantically equivalent ReCOGS from Wu et al 2023, which we built a model for earlier in RASP described at https://github.com/willy-b/RASP-for-ReCOGS ).

To try this out, run `python cogs_examples_in_rasp.py`!

`cogs_examples_in_rasp.py` can be used to evaluate on training data (default, use `--num_train_examples_to_check` to adjust the number of examples evaluated) 

or dev set data (`--use_dev_split`), 

or run a real test/gen evaluation (`--use_test_split`, `--use_gen_split`).

Context:
It is simpler and runs faster to just train a Transformer on examples!

This is an academic exercise, writing a neural network compatible program by hand in the Restricted Access Sequence Processing (compilable to Transformer) language (Weiss et al 2021, https://arxiv.org/abs/2106.06981 )
to prove a Transformer can perform a particular type of solution (building a systematic and compositional solution without a tree-structured/hierarchical representation (flat attention-head based pattern matching) that can handle recursive grammar by unrolling it in the decoder loop).

**We want to show a Transformer can solve the structural generalization splits of the "COGS: A Compositional Generalization Challenge Based on Semantic Interpretation"**
(Kim and Linzen 2020, https://aclanthology.org/2020.emnlp-main.731 ) task in the original COGS format - **Transformers trained from scratch on the training split have been reported in the literature to do as poorly as 0% accuracy on the structural generalizations by multiple authors.**

(We referenced the convenient summary of Kim, Linzen's COGS task grammar and vocab made by IBM's CPG project (their utilities, not their CPG project itself) at https://github.com/IBM/cpg/blob/c3626b4e03bfc681be2c2a5b23da0b48abe6f570/src/model/cogs_data.py#L523 during development, restricting ourselves to the vocabulary available in the training data.)

See also our finished RASP solution to the semantically equivalent ReCOGS task (Wu et al 2023, "ReCOGS: How Incidental Details of a Logical Form Overshadow an Evaluation of Semantic Interpretation", https://arxiv.org/abs/2303.13716) at https://github.com/willy-b/learning-rasp/blob/main/word-level-pos-tokens-recogs-style-decoder-loop.rasp 

with preprint of paper for the ReCOGS solution at https://arxiv.org/abs/2504.15349 (or https://raw.githubusercontent.com/willy-b/RASP-for-ReCOGS/main/rasp-for-recogs_pos-wbruns-2024-draft.pdf ) for context on this work.

Note this work is focused on structural generalization, not lexical generalization.

## Results

**Test set results:** `99.97% or 2999 out of 3000 string exact match (95% confidence interval: 99.81% to 100.00%)` 

(upper bound is less than 100% but greater than or equal to 99.995%, as 1 missed in 3000 could be consistent with an error rate less than or equal to 1 in 20,000 (99.995% exact match accuracy or better) within 95% confidence). 

**Generalization split results** (results are in for all splits except cp_recursion, which is still evaluating).

The focus of this work is on structural generalizations, especially those involving the recursive, potentially center-embedding prepositional phrases.

We find that **the main structural generalization split of interest** (was a focus of the separate RASP-for-ReCOGS paper in the ReCOGS variant of COGS), **obj_pp_to_subj_pp , had 100% string exact match performance**:
```
obj_pp_to_subj_pp: 100.00% (95.00% confidence interval: 99.63% to 100.00% (1000.0 out of 1000))
```

The other main structural generalization split, **prepositional phrase recursion also had strong performance (98.40% string exact match), despite RASP-for-COGS lacking any support for recursive rules or hierarchical representation** (just unrolling recursion in the decoder loop):
```
pp_recursion: 98.40% (95.00% confidence interval: 97.41% to 99.08% (984.0 out of 1000))
```

The other complete generalization splits for non-recursive parts of the COGS grammar are given below:
```
Exact match score on first 19000 of COGS gen:

99.96% or 18993.0 out of 19000 (95% confidence interval: 99.92% to 99.99%)

Exact Match % by category:
active_to_passive: 100.00% (95.00% confidence interval: 99.63% to 100.00% (1000.0 out of 1000))
do_dative_to_pp_dative: 99.90% (95.00% confidence interval: 99.44% to 100.00% (999.0 out of 1000))
obj_omitted_transitive_to_transitive: 100.00% (95.00% confidence interval: 99.63% to 100.00% (1000.0 out of 1000))
obj_pp_to_subj_pp: 100.00% (95.00% confidence interval: 99.63% to 100.00% (1000.0 out of 1000))
obj_to_subj_common: 100.00% (95.00% confidence interval: 99.63% to 100.00% (1000.0 out of 1000))
obj_to_subj_proper: 99.90% (95.00% confidence interval: 99.44% to 100.00% (999.0 out of 1000))
only_seen_as_transitive_subj_as_unacc_subj: 100.00% (95.00% confidence interval: 99.63% to 100.00% (1000.0 out of 1000))
only_seen_as_unacc_subj_as_obj_omitted_transitive_subj: 100.00% (95.00% confidence interval: 99.63% to 100.00% (1000.0 out of 1000))
only_seen_as_unacc_subj_as_unerg_subj: 100.00% (95.00% confidence interval: 99.63% to 100.00% (1000.0 out of 1000))
passive_to_active: 100.00% (95.00% confidence interval: 99.63% to 100.00% (1000.0 out of 1000))
pp_dative_to_do_dative: 99.90% (95.00% confidence interval: 99.44% to 100.00% (999.0 out of 1000))
prim_to_inf_arg: 100.00% (95.00% confidence interval: 99.63% to 100.00% (1000.0 out of 1000))
prim_to_obj_common: 99.80% (95.00% confidence interval: 99.28% to 99.98% (998.0 out of 1000))
prim_to_obj_proper: 100.00% (95.00% confidence interval: 99.63% to 100.00% (1000.0 out of 1000))
prim_to_subj_common: 100.00% (95.00% confidence interval: 99.63% to 100.00% (1000.0 out of 1000))
prim_to_subj_proper: 100.00% (95.00% confidence interval: 99.63% to 100.00% (1000.0 out of 1000))
subj_to_obj_common: 99.80% (95.00% confidence interval: 99.28% to 99.98% (998.0 out of 1000))
subj_to_obj_proper: 100.00% (95.00% confidence interval: 99.63% to 100.00% (1000.0 out of 1000))
unacc_to_transitive: 100.00% (95.00% confidence interval: 99.63% to 100.00% (1000.0 out of 1000))

[skip multiple blank lines]

Exact Match score on first 19000 of COGS gen:

99.96315789473684% or 18993.0 out of 19000 (95% confidence interval: 99.92% to 99.99%)
```

Lastly, the third and final structural generalization split, the sentential complement recursion (cp_recursion) split, alone is still being evaluated, but also has strong performance so far.
Here are the scores on the trials completed so far, out of what will be n=1000 total for the split:
```
Exact Match % by category (so far):
cp_recursion: 99.85% (95.00% confidence interval: 99.15% to 100.00% (651.0 out of 652))
```

## RASP-for-COGS Encoder-Decoder Schematic (simplified)

![](rasp-for-cogs-decoder-loop-figure-incl-encoder-and-decoder.png)

And Table 1 showing the 19 grammar patterns used above for noun-verb relationship ordering templates associated with specific official COGS training examples (of many examples of each in the training set), as well as the first center embedded prepositional phrase official COGS training example that motivates pp masking (of many):

![](rasp-for-cogs-grammar-patterns-matched-to-official-training-examples-adapted-from-wb-rasp-for-recogs-paper.svg)

Note the number, 19, of the grammar patterns coincidentally matches the number of non-recursive generalization splits but that is a coincidence and these are not derived from generalization set data (see examples in the Table).

## Intended use and warnings

These are intended be obvious or self-evident conditions, but especially for those who do not read the motivating RASP-for-ReCOGS paper ( https://arxiv.org/abs/2504.15349 ):

The RASP-for-COGS model applied to unintended use WILL give invalid results or halt - **we have NOT provided a general language model**, we have provided a simulation of how a Transformer could perform a specific task.
The RASP-for-COGS model/simulation as provided is **for research purposes only** to prove feasibility of the COGS task by Transformers (especially using a non-tree structured, non-hierarchical flat pattern matching and masking approach, unrolling recursion in the decoder loop) and is not appropriate for ANY other uses whatsoever without modification. For one, an actual Transformer performing the equivalent operations would run orders of magnitude faster, which should be reason enough to not want to use the RASP simulation for actual input-output tasks outside of a research setting. Fallback paths for out-of-grammar examples are also not provided or added by default so the RASP model may halt on natural inputs and will fail to perform semantic parsing on natural language text (would need to be modified simulating training on additional text). It can only run on the in-distribution (non-augmented) training data, and the dev, test, and gen sets (or any reasonable expected generalizations of the training data, not necessarily gen set provided) of COGS, though such aspects could be added. We provide the code for reproducing the results of this study and for researchers who are capable of writing RASP themselves to build upon the work and/or more easily apply RASP to their own problems given our examples, not for immediate application to any other tasks without appropriate modification.
