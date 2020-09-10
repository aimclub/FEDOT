Components
==========

Main logical and structural blocks of the FEDOT


The FEDOT framework includes the set of components: - Core, that allows
to generate and use the composite model for the modeling tasks; -
Utilities, that solves the supplementary tasks (benchmarks generation,
data preprocessing, etc); - Benchmarks, that provides the interface for
the comparison of Fedot with the baseline and external state-of-the-art
solutions (the TPOT, H2O, AutoKeras, MLBox are supported now); -
Examples, that illustrates the main features of the framework; -
Real-world cases, that represents the scenarios of the adaptation of
Fedot to the challenging problems from different domains; - Test, that
allows verifying the correctness of the existing classes and functions;

In the framework, the representation of the composite model consist of
several concepts:

-  Atomic model - the model of target process, which transforms input
   data in the result of modeling Y=F_mod(X) and is considered
   undecomposable in the course of solving a specific task of
   identification of composite models. At the same time, the model can
   reproduce all the scales of variability Y, as well as only a part of
   them;
-  Operation model - the model that transformation of one or more data
   sets (modeling results) into a new, more qualitative one: Yn+1=
   F_op(Y1…Yn);
-  Composite model is a composite model that represents a chain of all
   atomic models identified during the application of the evolutionary
   algorithm;
-  Atomization - the transformation of a composite model into an atomic
   model for use in subsequent launches of the framework.

There is the main classes of the FEDOT:

**Chain** - encapsulates graph of composite model from operations;
allows to calculate the whole chain of models recursively;

**Node** - chain component that contains the model, references to parent
and child nodes, and cached model object;

**NodeFactory** - a factory for creating a necessary Node object;

**EvaluationStrategy** - strategy of model learning and calculation of
predicted values.