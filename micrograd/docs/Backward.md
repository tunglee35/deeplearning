# Explanation on how the backward function works

### Install micrograd to local dir
```bash
pip install --target=$PWD micrograd
```

## Concepts
* Loss value: a value computed from a function with data and weight tensors as inputs

## Examine engine.py
#### Data structure
* ...
* prev: link to the children nodes (this create a massive graph of nodes)
* grad (soul of the algorithm): represent the rate at which the output change with respect to the current value
  * In a neural net, there is one loss value and multiple inputs/weights tensors. At any given moment in time, we compute the loss value and change the weights to get smaller loss value.
  * => Knowing the rate of change of the loss value compare to other weights is a very important information

#### __backward__ function
* To: identify which node has the most impact on the loss value
* Behavior: calculate the gradient of the tensor with respect to its children nodes (e.g if a + b = c, then a and b are the children nodes of c)
* Internal:
  1. Build a topological sort of the graph, with the current value as root
  2. Change its grad value to 1 (because the gradient of the current value with respect to itself is 1)
  3. For each children node in the topological sort, call its _backward function (once this is called, you can identify which node has the most impact on the loss value - a critical piece of information for training a neural net)

#### __add__(self, other) function
* To: perform normal addition
* Behavior: add two tensors together
* Side effect: replace the _backward function of the result tensor with a function that can calculate the gradient of the result tensor with respect to the two input tensors

  * Why the _backward function of the result tensor (c), add to the grad of its operands (a and b) by c.grad?
    * Because the gradient of c with respect to a is 1, and the gradient of c with respect to b is 1
    * So the gradient of c with respect to a and b is 1
    * So the gradient of c with respect to a and b is c.grad


* The same pattern can be applied to other operators