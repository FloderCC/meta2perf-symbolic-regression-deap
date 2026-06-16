# src/utils/gp_custom_operators.py

import random
import sys
from collections import defaultdict
from itertools import product

from deap import gp


def cxOnePoint_limited(ind1, ind2, max_nodes, max_height):
    """
    One-point crossover with hard node-count and height limits.

    It builds all compatible crossover-point pairs, shuffles them, and applies
    the first pair that keeps both offspring within max_nodes and max_height.
    """

    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    types1 = defaultdict(list)
    types2 = defaultdict(list)

    for idx, node in enumerate(ind1[1:], 1):
        types1[node.ret].append(idx)

    for idx, node in enumerate(ind2[1:], 1):
        types2[node.ret].append(idx)

    common_types = list(set(types1.keys()).intersection(types2.keys()))

    if not common_types:
        return ind1, ind2

    candidate_pairs = []

    for type_ in common_types:
        candidate_pairs.extend(product(types1[type_], types2[type_]))

    random.shuffle(candidate_pairs)

    for index1, index2 in candidate_pairs:
        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)

        size_subtree1 = slice1.stop - slice1.start
        size_subtree2 = slice2.stop - slice2.start

        new_size1 = len(ind1) - size_subtree1 + size_subtree2
        new_size2 = len(ind2) - size_subtree2 + size_subtree1

        if new_size1 > max_nodes or new_size2 > max_nodes:
            continue

        child1 = gp.PrimitiveTree(ind1[:])
        child2 = gp.PrimitiveTree(ind2[:])

        child1[slice1] = ind2[slice2]
        child2[slice2] = ind1[slice1]

        if child1.height > max_height or child2.height > max_height:
            continue

        subtree1 = ind1[slice1]
        subtree2 = ind2[slice2]

        ind1[slice1] = subtree2
        ind2[slice2] = subtree1

        return ind1, ind2

    return ind1, ind2


def _choose_terminal(pset, type_):
    """
    Selects a terminal of the requested type and instantiates it when it is
    an ephemeral constant.
    """

    try:
        term = random.choice(pset.terminals[type_])
    except IndexError:
        _, _, traceback = sys.exc_info()
        raise IndexError(
            "generate_limited tried to add a terminal of type '%s', "
            "but there is none available." % (type_,)
        ).with_traceback(traceback)

    if type(term) is gp.MetaEphemeral:
        term = term()

    return term


def _primitive_fits_minimum_budget(prim, expr_len, stack_len, max_len):
    """
    Checks whether adding this primitive leaves at least one node available
    for each pending branch.

    After adding the primitive:
    - 1 node is consumed by the primitive itself;
    - prim.arity new children are pushed to the stack;
    - stack_len pending nodes were already waiting.

    Therefore, a conservative feasibility check is:

        expr_len + 1 + stack_len + prim.arity <= max_len
    """

    return expr_len + 1 + stack_len + prim.arity <= max_len


def generate_limited(pset, min_, max_, condition, max_len, type_=None):
    """
    Similar to deap.gp.generate, but with hard limits on both:

    - maximum height: controlled by max_;
    - maximum number of nodes: controlled by max_len.

    The generated expression is guaranteed to satisfy:

        len(expr) <= max_len
        gp.PrimitiveTree(expr).height <= max_

    provided the primitive set has valid terminals for the required types.
    """

    if max_len < 1:
        raise ValueError("max_len must be at least 1.")

    if min_ < 0:
        min_ = 0

    if max_ < 0:
        max_ = 0

    if min_ > max_:
        min_ = max_

    if type_ is None:
        type_ = pset.ret

    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]

    while stack:
        if len(expr) >= max_len:
            raise ValueError("Cannot complete tree within max_len.")

        depth, current_type = stack.pop()

        remaining_slots = len(stack)
        remaining_nodes = max_len - len(expr)

        # Need at least one node for the current item and one node for
        # every item already pending in the stack.
        must_use_terminal_by_size = remaining_nodes <= remaining_slots + 1

        # Hard height limit. If the current depth reaches the selected
        # target height, this node must be terminal. This is what prevents
        # height from becoming max_ + 1, max_ + 2, etc.
        must_use_terminal_by_height = depth >= height

        if (
            must_use_terminal_by_height
            or condition(height, depth)
            or must_use_terminal_by_size
        ):
            expr.append(_choose_terminal(pset, current_type))
            continue

        primitives = [
            prim
            for prim in pset.primitives[current_type]
            if _primitive_fits_minimum_budget(
                prim=prim,
                expr_len=len(expr),
                stack_len=len(stack),
                max_len=max_len,
            )
        ]

        if not primitives:
            expr.append(_choose_terminal(pset, current_type))
            continue

        prim = random.choice(primitives)
        expr.append(prim)

        for arg in reversed(prim.args):
            stack.append((depth + 1, arg))

    return expr


def genFull_limited(pset, min_, max_, max_len, type_=None):
    """
    Size-limited version of deap.gp.genFull.

    It tries to behave like genFull, but if the node budget becomes too tight,
    terminals are forced to respect max_len.
    """

    def condition(height, depth):
        return depth == height

    return generate_limited(
        pset=pset,
        min_=min_,
        max_=max_,
        condition=condition,
        max_len=max_len,
        type_=type_,
    )


def genGrow_limited(pset, min_, max_, max_len, type_=None):
    """
    Size-limited version of deap.gp.genGrow.
    """

    def condition(height, depth):
        return depth == height or (
            depth >= min_ and random.random() < pset.terminalRatio
        )

    return generate_limited(
        pset=pset,
        min_=min_,
        max_=max_,
        condition=condition,
        max_len=max_len,
        type_=type_,
    )


def genHalfAndHalf_limited(pset, min_, max_, max_len, type_=None):
    """
    Size-limited version of deap.gp.genHalfAndHalf.

    Half the time it uses genGrow_limited; the other half it uses
    genFull_limited.
    """

    method = random.choice((genGrow_limited, genFull_limited))

    return method(
        pset=pset,
        min_=min_,
        max_=max_,
        max_len=max_len,
        type_=type_,
    )