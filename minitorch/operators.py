"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, TypeVar

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# Forward functions
def mul(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def id(a: float) -> float:
    """Identity function: returns the input unchanged."""
    return a


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def neg(a: float) -> float:
    """Negate the input number."""
    return -a


def lt(a: float, b: float) -> bool:
    """Return True if a < b, else False."""
    return a < b


def eq(a: float, b: float) -> bool:
    """Return True if a == b, else False (strict equality)."""
    return a == b


def max(a: float, b: float) -> float:
    """Return the maximum of a and b."""
    return a if a > b else b


def is_close(a: float, b: float, tol: float = 1e-2) -> bool:
    """Return True if a and b are within tol of each other."""
    return abs(a - b) < tol


def sigmoid(x: float) -> float:
    """Compute the sigmoid function in a numerically stable way.

    For a >= 0: 1 / (1 + exp(-a))
    For a < 0: exp(a) / (1 + exp(a))
    """
    return 1.0 / (1.0 + math.exp(-x))


def relu(a: float) -> float:
    """Compute the ReLU function: max(0, a)."""
    return a if a > 0 else 0.0


def log(a: float) -> float:
    """Compute the natural logarithm of a."""
    return math.log(a)


def exp(a: float) -> float:
    """Compute the exponential of a (e^a)."""
    return math.exp(a)


def inv(a: float) -> float:
    """Compute the multiplicative inverse (1/a) of a."""
    return 1.0 / a


# Backward (gradient) functions
def log_back(a: float, grad: float) -> float:
    """Compute the gradient of log(a) w.r.t a for backpropagation.

    grad: upstream gradient
    returns: grad / a
    """
    return grad / a


def inv_back(a: float, grad: float) -> float:
    """Compute the gradient of 1/a w.r.t a for backpropagation.

    grad: upstream gradient
    returns: -grad / (a^2)
    """
    return -grad / (a * a)


def relu_back(a: float, grad: float) -> float:
    """Compute the gradient of ReLU w.r.t a for backpropagation.

    grad: upstream gradient
    returns: grad if a > 0 else 0
    """
    return grad if a > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

T = TypeVar("T", int, float)
U = TypeVar("U")
R = TypeVar("R")


def map(func: Callable[[T], T], items: Iterable[T]) -> list[T]:
    """Apply `func` to each element in `items` and return a list of results."""
    return [func(x) for x in items]


def zipWith(
    item1: Iterable[T], item2: Iterable[T], func: Callable[[T, T], U]
) -> list[U]:
    """Higher-order function that combines elements from two iterables using a given function.

    Args:
    ----
        item1: First iterable of elements of type T.
        item2: Second iterable of elements of type T.
        func: A function that takes two arguments (one from each iterable) and returns type U.

    Returns:
    -------
        A list of results of type U.

    """
    return [func(a, b) for a, b in zip(item1, item2)]


def reduce(fn: Callable[[R, T], R], start: R) -> Callable[[Iterable[T]], R]:
    r"""Higher-order reduce.

    Args:
    ----
        fn: A two-argument function to combine two values.
        start: Initial accumulator value.

    Returns:
    -------
        A function that takes a list (or iterable) `ls` of elements and computes the reduction:
        fn(x_n, fn(x_{n-1}, ..., fn(x_1, start)...))

    """

    def _reduce(ls: Iterable[T], fn: Callable[[R, T], R], start: R) -> R:
        accumulator = start
        for i in ls:
            accumulator = fn(accumulator, i)
        return accumulator

    return lambda ls: _reduce(ls, fn, start)


def negList(lst: list[T]) -> list[T]:
    """Negate all elements in a list using `map`."""
    return map(lambda x: -x, lst)


def addLists(lst1: list[T], lst2: list[T]) -> list[float]:
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(lst1, lst2, add)


def sum(lst: list[T]) -> float:
    """Sum all elements in a list using reduce"""
    return reduce(add, 0)(lst)


def prod(lst: list[T]) -> float:
    """Calculate the product of all elements in a list using reduce"""
    return reduce(mul, 1)(lst)
