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

with draft paper for the ReCOGS solution at https://raw.githubusercontent.com/willy-b/RASP-for-ReCOGS/main/rasp-for-recogs_pos-wbruns-2024-draft.pdf for context on this work.

Note this work is focused on structural generalization, not lexical generalization.

## Results

Test set results: `99.97% or 2999 out of 3000 string exact match (95% confidence interval: 99.81% to 100.00%)` 

(upper bound is less than 100% but greater than or equal to 99.995%, as 1 missed in 3000 could be consistent with an error rate less than or equal to 1 in 20,000 (99.995% exact match accuracy or better) within 95% confidence). 

Generalization split results (results are in for all splits except cp_recursion, pp_recursion, which are still evaluating):
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

Note that the main structural generalization split of interest (was a focus of the separate RASP-for-ReCOGS paper in the ReCOGS variant of COGS), obj_pp_to_subj_pp , had 100% string exact match performance:
```
obj_pp_to_subj_pp: 100.00% (95.00% confidence interval: 99.63% to 100.00% (1000.0 out of 1000))
```

Sentential complement recursion (cp_recursion) and prepositional phrase recursion (pp_recursion) splits are still being evaluated.
Here are the scores on the trials completed so far, out of what will be n=1000 total for each split:
```
Exact Match % by category (so far):
cp_recursion: 100.00% (95.00% confidence interval: 98.39% to 100.00% (227.0 out of 227))
pp_recursion: 99.06% (95.00% confidence interval: 97.60% to 99.74% (420.0 out of 424))
```
