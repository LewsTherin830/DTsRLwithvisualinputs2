# DecisionTreeLibrary
![coverage](https://gitlab.com/leocus/decisiontreelibrary/badges/master/coverage.svg)
This library aims to be a general library for building/using decision trees.
In fact, with this library we do not aim to manage only the cases in which the DT is used for classification or regression purposes, but we make it work also for more recent applications, e.g. Reinforcement Learning.
The library has 3 main entities:
- Leaves: These classes learn a subset -> action mapping (e.g. subset -> class in classification, subset -> value in regression, state -> action in RL)
- Conditions: These classes split their input space into two subsets. The type of conditions is not bound to any type (e.g. orthogonal, oblique, etc), in fact, a class extending Condition can implement its own splitting criterion.
- Trees: These classes are "wrappers" that contain a root (either a leaf or a condition) and implement methods to manage the tree.
