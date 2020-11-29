# Loss functions

This document explains some of the loss functions that have been provided,
especially how gradients have been computed.

In this document, we use `y` to represent the ground truth vector of length `n`
and `y_hat` to represent the output of the last layer, also of length `n`.

## MSE Loss

## Softmax Loss

### Special Case: 1-Hot Expectation

This is a special case where we expect `y` to be of the form `(0, 0, ..., 0, 1, 0, ..., 0, 0)`, i.e.
a single `y[i]` has value 1 and all other `y[j]` for `j != i` have value 0.

Let's start with the definition of the softmax function for the `i`'th output:

```
softmax(y_hat, i) = exp(y_hat_i) / sum_k(exp(y_hat_k))  [eq 0]
```

We want to maximize the probability `p(y, y_hat) = softmax(y_hat, i)`.
In order to make sure the gradient is [well scaled](https://stats.stackexchange.com/questions/174481/why-to-optimize-max-log-probability-instead-of-probability),
we will instead minimize the negative of the log probability.
Without any regularization terms, the loss function is simply:

```
L(y, y_hat) = -ln(p(y, y_hat))
            = -ln(softmax(y_hat, i))
            = -ln(exp(y_hat_i) / sum_k(exp(y_hat_k)))
```

Using the logarithm identity [ln(x/y) = ln(x) - ln(y)](https://en.wikipedia.org/wiki/List_of_logarithmic_identities#Using_simpler_operations):

```
L(y, y_hat) = -ln(exp(y_hat_i)) + ln(sum_k(exp(y_hat_k)))
```

Let's compute the gradient of `L` w.r.t. output unit `j`:

```
d(L, y_hat_j) = d(-ln(exp(y_hat_i)) + ln(sum_k(exp(y_hat_k))), y_hat_j)
```

First we use the [rule of linearity](https://en.wikipedia.org/wiki/Linearity_of_differentiation):

```
d(L, y_hat_j) = -d(ln(exp(y_hat_i)), y_hat_j) + d(ln(sum_k(exp(y_hat_k))), y_hat_j)
```

The [derivative of `ln(f(x))`](https://en.wikipedia.org/wiki/Logarithm#Derivative_and_antiderivative) is `f'(x)/f(x)`:

```
d(L, y_hat_j) = -d(exp(y_hat_i), y_hat_j) / exp(y_hat_i) + d(sum_k(exp(y_hat_k)), y_hat_j) / sum_k(exp(y_hat_k))
```

The [derivative of exp(x)](https://en.wikipedia.org/wiki/Derivative#Rules_for_basic_functions) is `exp(x)`
the [derivative of a constant](https://en.wikipedia.org/wiki/Derivative#Rules_for_combined_functions) is 0, and again using the rule of linearity:

```
d(L, y_hat_j) = -d(exp(y_hat_i), y_hat_j) / exp(y_hat_i) + exp(y_hat_j) / sum_k(exp(y_hat_k))
```

There are two cases: `i = j` and `i != j`.
We note that `exp(y_hat_i)` is constant if `i != j` (and therefore its derivative is 0); and it is `exp(y_hat_i)` if `i = j`.
We also note that the second term `exp(y_hat_j) / sum_k(exp(y_hat_k))` is the definition of `softmax(y_hat, j)`.
Therefore:

```
d(L, y_hat_j) = { -1 + softmax(y_hat, j)   if i = j
                { softmax(y_hat, j)        if i != j
```

### General Case

```
DRAFT!!

The softmax loss function:
[ 0] softmax(y_hat, i) = exp(y_hat_i) / sum_k(exp(y_hat_k))  [definition of softmax]
[ 1] p(y, y_hat) = y dot softmax(y_hat)
[ 2] L(y, y_hat) = -ln(p(y, y_hat))
[ 3]             = -ln(y dot softmax(y_hat))
[ 4]             = -ln(sum_j(y_j * softmax(y_hat, i)))
[ 5]             = -ln(sum_j(y_j * exp(y_hat_j) / sum_k(y_k * exp(y_hat_k))))  [0 into 4]

The derivative of the softmax loss function w.r.t. y_hat_i:
[ 6] d(L, y_hat_i) = d(-ln(sum_j(y_j * exp(y_hat_j) / sum_k(y_k * exp(y_hat_k))))), y_hat_i)
[ 7]               = d(-ln(sum_j(y_j * exp(y_hat_j)) / sum_k(y_k * exp(y_hat_k)))), y_hat_i)  [a/d + b/d) == (a+b)/d]
[ 8]               = d(-ln(sum_j(y_j * exp(y_hat_j))) + ln(sum_k(y_k * exp(y_hat_k))), y_hat_i)  [ln(a/b) == ln(a) - ln(b)]
[ 9]               = -d(ln(sum_j(y_j * exp(y_hat_j))), y_hat_i) + d(ln(sum_k(y_k * exp(y_hat_k))), y_hat_i)  [rule of linearity]
[10]               = -d(sum_j(y_j * exp(y_hat_j)), y_hat_i)) / sum_j(y_j * exp(y_hat_j)) + d(sum_k(y_k * exp(y_hat_k)), y_hat_i) / sum_k(y_k * exp(y_hat_k))  [d(ln(f(x)), x) == d(f(x), x) / f(x)]
[11]               = -d(y_i * exp(y_hat_i), y_hat_i) / sum_j(y_j * exp(y_hat_j)) + d(y_i * exp(y_hat_i), y_hat_i) / sum_k(y_k * exp(y_hat_k))  [rule of linearity, constant rule]
[12]               =                                                             + y_i * exp(y_hat_i) / sum_k(y_k * exp(y_hat_k))  [rule of linearity, d(exp(x), x) == exp(x)]
[13]               =                                                             + y_i / sum(y) * softmax(y_hat, i)  [?]
```
