# Naive Bayes Classifier

### Introduction
Bayesian models work on the premise that given evidence, E, we can infer a hypothesis, H.

### Steps:

1. Understanding dataset and creating 7 distinguishable features.
2. The intuition for including "has_no_evidence" is informed by the fact that base cases should be accounted for. This however exposes a Bayesian flaw, which is that it performs poorly in an unbalanced class situation.
3. Having generated our features, calculate the prior, P(H) and likelihood ratio P(E|H)/P(E)
4. Next, calculate class probabilities.
5. Finally, return maximum probability class as prediction.

How to run:
Python NaiveBayes.py <insert path to dataset>

### NOTE: Dataset was transformed from a single input-single output match dataset
