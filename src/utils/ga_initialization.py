import heapq
import random
from typing import List, Optional, Tuple


class SrVecParser:
    """
    Vector-to-expression parser for symbolic regression.

    Representation
    --------------
    The chromosome is a fixed-length list of prefix tokens. Each token uses 5 genes:

        [node_type, var_code, const_value, unary_code, binary_code]

    node_type:
        1 -> VAR    (variable leaf)
        2 -> CONST  (constant leaf)
        3 -> UNARY  (one child)
        4 -> BINARY (two children)

    var_code:
        - index into var_list

    const_value:
        - used only by CONST nodes

    unary_code:
        - index into unary_operators

    binary_code:
        - index into binary_operators

    Notes
    -----
    * Every in-bounds vector decodes to a syntactically valid SR expression.
    * max_depth is a hard upper bound, but equations are usually much smaller because
      projected leaf nodes terminate the expansion early.
    * max_tokens bounds the prefix stream length and keeps the chromosome fixed-size,
      which makes it compatible with vector-based initializers such as OBLESA.
    """

    NODE_VAR = 1
    NODE_CONST = 2
    NODE_UNARY = 3
    NODE_BINARY = 4

    GENES_PER_TOKEN = 5

    def __init__(
        self,
        var_list: List[str],
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
        max_depth: int = 35,
        max_tokens: Optional[int] = None,
    ):
        self.var_list = var_list
        self.binary_operators = binary_operators or ['add', 'sub', 'mul', 'div']
        self.unary_operators = [u for u in (unary_operators or ['sqrt', 'exp', 'log']) if u != 'drop']
        self.max_depth = max(1, int(max_depth))
        self.max_tokens = int(max_tokens) if max_tokens is not None else max(63, 4 * self.max_depth + 1)

        # Depth-aware probabilities used only by the balanced random generator.
        # The order is always: [VAR, CONST, UNARY, BINARY].
        # Order: [VAR, CONST, UNARY, BINARY]
        self.root_node_probs = [0.02, 0.02, 0.21, 0.75]
        self.shallow_node_probs = [0.06, 0.06, 0.23, 0.65]
        self.middle_node_probs = [0.15, 0.15, 0.20, 0.50]
        self.deep_node_probs = [0.35, 0.35, 0.18, 0.12]

        # At the maximum allowed depth, only leaves are permitted.
        # The order is: [VAR, CONST].
        self.max_depth_leaf_probs = [0.50, 0.50]

    def get_bounds_list(self, const_min: float = -1.0, const_max: float = 5.0) -> list:
        """Generate bounds for the fixed-length chromosome."""
        bounds_list = []
        max_var = max(len(self.var_list) - 1, 0)
        max_unary = max(len(self.unary_operators) - 1, 0)
        max_binary = max(len(self.binary_operators) - 1, 0)

        for _ in range(self.max_tokens):
            bounds_list.append((1, 4))
            bounds_list.append((0, max_var))
            bounds_list.append((const_min, const_max))
            bounds_list.append((0, max_unary))
            bounds_list.append((0, max_binary))
        return bounds_list

    def encode_equation(self, equation_str: str) -> list:
        """Encode a nested DEAP-style functional expression into the fixed-length prefix vector."""
        import ast

        eq = (equation_str or '').strip()
        if not eq:
            raise ValueError('equation_str must be a non-empty string.')

        try:
            parsed = ast.parse(eq, mode='eval')
        except SyntaxError as exc:
            raise ValueError(f'Invalid equation syntax: {exc}') from exc

        var_to_idx = {name: idx for idx, name in enumerate(self.var_list)}
        unary_to_idx = {name: idx for idx, name in enumerate(self.unary_operators)}
        binary_to_idx = {name: idx for idx, name in enumerate(self.binary_operators)}

        def numeric_constant(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                inner = numeric_constant(node.operand)
                if inner is not None:
                    return -float(inner)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
                inner = numeric_constant(node.operand)
                if inner is not None:
                    return float(inner)
            return None

        def build_tokens(node, depth):
            if depth >= self.max_depth:
                raise ValueError(
                    f'Equation depth exceeds max_depth={self.max_depth}. '
                    f'Increase max_depth to encode this expression.'
                )

            if isinstance(node, ast.Name):
                if node.id not in var_to_idx:
                    raise ValueError(f"Unknown variable '{node.id}'. Allowed variables: {self.var_list}")
                return [[self.NODE_VAR, var_to_idx[node.id], 0.0, 0, 0]]

            const_value = numeric_constant(node)
            if const_value is not None:
                return [[self.NODE_CONST, 0, float(const_value), 0, 0]]

            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError('Only simple function names are supported in expressions.')
                op_name = node.func.id

                if op_name in unary_to_idx:
                    if len(node.args) != 1:
                        raise ValueError(f"Unary operator '{op_name}' expects exactly 1 argument.")
                    child_tokens = build_tokens(node.args[0], depth + 1)
                    return [[self.NODE_UNARY, 0, 0.0, unary_to_idx[op_name], 0]] + child_tokens

                if op_name in binary_to_idx:
                    if len(node.args) != 2:
                        raise ValueError(f"Binary operator '{op_name}' expects exactly 2 arguments.")
                    left_tokens = build_tokens(node.args[0], depth + 1)
                    right_tokens = build_tokens(node.args[1], depth + 1)
                    return [[self.NODE_BINARY, 0, 0.0, 0, binary_to_idx[op_name]]] + left_tokens + right_tokens

                raise ValueError(
                    f"Unknown operator '{op_name}'. Allowed unary operators: {self.unary_operators}. "
                    f"Allowed binary operators: {self.binary_operators}."
                )

            raise ValueError(
                f'Unsupported expression node: {type(node).__name__}. '
                'Use only variables, numeric constants, and functional operators.'
            )

        tokens = build_tokens(parsed.body, depth=0)
        if len(tokens) > self.max_tokens:
            raise ValueError(
                f'Equation needs {len(tokens)} tokens but max_tokens={self.max_tokens}. '
                'Increase max_tokens to encode this expression.'
            )

        tokens.extend([[self.NODE_VAR, 0, 0.0, 0, 0] for _ in range(self.max_tokens - len(tokens))])

        flat = []
        for token in tokens:
            flat.extend(token)
        return flat

    def _format_const(self, value: float) -> str:
        if abs(value) < 1e-12:
            value = 0.0
        return repr(float(value))

    def _sanitize_type(self, value: float) -> int:
        return max(1, min(4, int(round(value))))

    def _sanitize_index(self, value: float, length: int) -> int:
        if length <= 0:
            return 0
        return max(0, min(length - 1, int(round(value))))

    def _fallback_leaf(self, var_code: int, const_value: float, prefer_const: bool = False) -> str:
        if prefer_const or not self.var_list:
            return self._format_const(const_value)
        return self.var_list[var_code % len(self.var_list)]

    def _project_node_type(
        self,
        requested_type: int,
        *,
        depth: int,
        remaining_tokens: int,
        open_slots: int,
        is_root: bool,
    ) -> int:
        """Project a raw node-type gene to a structurally valid node type."""
        allowed = []

        if (
                depth < self.max_depth - 1
                and self.unary_operators
                and remaining_tokens >= open_slots
        ):
            allowed.append(self.NODE_UNARY)
        # For a binary node, after consuming this token we will need to close open_slots + 1.
        # That requires at least open_slots + 1 remaining tokens in the worst case.
        if depth < self.max_depth - 1 and self.binary_operators and remaining_tokens >= (open_slots + 1):
            allowed.append(self.NODE_BINARY)

        allowed.extend([self.NODE_VAR, self.NODE_CONST])

        if requested_type in allowed:
            return requested_type

        if is_root:
            if self.NODE_BINARY in allowed:
                return self.NODE_BINARY
            if self.NODE_UNARY in allowed:
                return self.NODE_UNARY
        return self.NODE_VAR if self.var_list else self.NODE_CONST

    def _decode_expr_slots(
        self,
        tokens: List[Tuple[int, int, int, int, float]],
        pos: int,
        depth: int,
        open_slots: int,
    ) -> Tuple[str, int, int]:
        if pos >= len(tokens):
            return self._fallback_leaf(0, 0.0), pos, max(open_slots - 1, 0)

        raw_type, var_code, const_value, unary_code, binary_code = tokens[pos]
        next_pos = pos + 1
        remaining_tokens = len(tokens) - next_pos
        node_type = self._project_node_type(
            raw_type,
            depth=depth,
            remaining_tokens=remaining_tokens,
            open_slots=open_slots,
            is_root=(pos == 0),
        )

        if node_type == self.NODE_VAR:
            return self.var_list[var_code % len(self.var_list)], next_pos, open_slots - 1

        if node_type == self.NODE_CONST:
            return self._format_const(const_value), next_pos, open_slots - 1

        if node_type == self.NODE_UNARY:
            op = self.unary_operators[unary_code % len(self.unary_operators)]
            child_expr, consumed_pos, open_after_child = self._decode_expr_slots(
                tokens, next_pos, depth + 1, open_slots
            )
            return f"{op}({child_expr})", consumed_pos, open_after_child

        op = self.binary_operators[binary_code % len(self.binary_operators)]
        left_expr, pos_after_left, open_after_left = self._decode_expr_slots(
            tokens, next_pos, depth + 1, open_slots + 1
        )
        right_expr, pos_after_right, open_after_right = self._decode_expr_slots(
            tokens, pos_after_left, depth + 1, open_after_left
        )
        return f"{op}({left_expr}, {right_expr})", pos_after_right, open_after_right

    def validate_vector_structure(self, encoded_genes: list) -> dict:
        """Validate how an arbitrary in-bounds vector is structurally projected."""
        if len(encoded_genes) % self.GENES_PER_TOKEN != 0:
            raise ValueError('Encoded genes list must have a length that is a multiple of 5.')

        tokens = []
        for i in range(0, len(encoded_genes), self.GENES_PER_TOKEN):
            node_type = self._sanitize_type(encoded_genes[i])
            var_code = self._sanitize_index(encoded_genes[i + 1], len(self.var_list))
            const_value = float(encoded_genes[i + 2])
            unary_code = self._sanitize_index(encoded_genes[i + 3], len(self.unary_operators))
            binary_code = self._sanitize_index(encoded_genes[i + 4], len(self.binary_operators))
            tokens.append((node_type, var_code, const_value, unary_code, binary_code))

        expr, end_pos, open_slots = self._decode_expr_slots(tokens, pos=0, depth=0, open_slots=1)
        return {
            'valid': open_slots == 0,
            'consumed_tokens': end_pos,
            'total_tokens': len(tokens),
            'ignored_suffix_tokens': max(len(tokens) - end_pos, 0),
            'open_slots_after_decode': open_slots,
            'expression': expr,
        }

    def decode_equation(self, encoded_genes: list) -> str:
        """Decode the chromosome into a DEAP-style functional expression."""
        if len(encoded_genes) % self.GENES_PER_TOKEN != 0:
            raise ValueError('Encoded genes list must have a length that is a multiple of 5.')

        tokens = []
        for i in range(0, len(encoded_genes), self.GENES_PER_TOKEN):
            node_type = self._sanitize_type(encoded_genes[i])
            var_code = self._sanitize_index(encoded_genes[i + 1], len(self.var_list))
            const_value = float(encoded_genes[i + 2])
            unary_code = self._sanitize_index(encoded_genes[i + 3], len(self.unary_operators))
            binary_code = self._sanitize_index(encoded_genes[i + 4], len(self.binary_operators))
            tokens.append((node_type, var_code, const_value, unary_code, binary_code))

        expr, _, _ = self._decode_expr_slots(tokens, pos=0, depth=0, open_slots=1)
        return expr or (self.var_list[0] if self.var_list else '0.0')

    def parse_srt_eq_to_deap_representation(self, equation_str: str) -> str:
        eq = (equation_str or '').strip()
        return eq if eq else (self.var_list[0] if self.var_list else '0.0')

    def canonicalize_expression(self, equation_str: str) -> str:
        """Lightweight canonicalization used for deduplication."""
        eq = (equation_str or '').replace(' ', '')
        if not eq:
            return self.var_list[0] if self.var_list else '0.0'

        def split_args(s: str) -> Tuple[str, str]:
            depth = 0
            for i, ch in enumerate(s):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                elif ch == ',' and depth == 0:
                    return s[:i], s[i + 1:]
            raise ValueError(f'Could not split binary arguments in: {s}')

        def canon(expr: str) -> str:
            expr = expr.strip()
            if not expr:
                return self.var_list[0] if self.var_list else '0.0'
            if '(' not in expr or not expr.endswith(')'):
                return expr

            op_end = expr.find('(')
            if op_end <= 0:
                return expr
            op = expr[:op_end]
            inside = expr[op_end + 1:-1]

            if op in self.unary_operators:
                return f"{op}({canon(inside)})"

            if op in self.binary_operators:
                left, right = split_args(inside)
                left_c = canon(left)
                right_c = canon(right)
                if op in {'add', 'mul'} and right_c < left_c:
                    left_c, right_c = right_c, left_c
                return f"{op}({left_c},{right_c})"

            return expr

        return canon(eq)

    def _choose_node_type(self, depth: int) -> int:
        """
        Depth-dependent random node selection.

        This keeps the current 5-gene token encoding, but makes the random
        generator less biased toward very small trees. The objective is to
        obtain a more balanced initial population containing small, medium,
        and large trees.

        Notes
        -----
        * At the root and shallow levels, operators are favored to encourage
          tree growth.
        * At middle levels, leaves and operators are more balanced.
        * At deeper levels, leaves are favored to close branches safely.
        * At max_depth - 1, only leaves are allowed.
        """
        if depth >= self.max_depth - 1:
            return random.choices(
                [self.NODE_VAR, self.NODE_CONST],
                weights=self.max_depth_leaf_probs,
                k=1,
            )[0]

        if depth == 0:
            probs = self.root_node_probs
        elif depth < self.max_depth * 0.25:
            probs = self.shallow_node_probs
        elif depth < self.max_depth * 0.60:
            probs = self.middle_node_probs
        else:
            probs = self.deep_node_probs

        return random.choices(
            [self.NODE_VAR, self.NODE_CONST, self.NODE_UNARY, self.NODE_BINARY],
            weights=probs,
            k=1,
        )[0]

    def random_individual(self, const_min: float = -1.0, const_max: float = 5.0) -> list:
        """Generate a structurally valid prefix chromosome."""
        tokens = [[self.NODE_VAR, 0, 0.0, 0, 0] for _ in range(self.max_tokens)]

        def fill(pos: int, depth: int, open_slots: int) -> Tuple[int, int]:
            if pos >= self.max_tokens or open_slots <= 0:
                return pos, open_slots

            remaining_tokens = self.max_tokens - (pos + 1)
            requested_type = self._choose_node_type(depth)
            node_type = self._project_node_type(
                requested_type,
                depth=depth,
                remaining_tokens=remaining_tokens,
                open_slots=open_slots,
                is_root=(pos == 0),
            )

            var_code = random.randint(0, len(self.var_list) - 1) if self.var_list else 0
            unary_code = random.randint(0, len(self.unary_operators) - 1) if self.unary_operators else 0
            binary_code = random.randint(0, len(self.binary_operators) - 1) if self.binary_operators else 0
            const_value = random.uniform(const_min, const_max)
            tokens[pos] = [node_type, var_code, const_value, unary_code, binary_code]
            next_pos = pos + 1

            if node_type in (self.NODE_VAR, self.NODE_CONST):
                return next_pos, open_slots - 1
            if node_type == self.NODE_UNARY:
                return fill(next_pos, depth + 1, open_slots)

            next_pos, open_slots = fill(next_pos, depth + 1, open_slots + 1)
            next_pos, open_slots = fill(next_pos, depth + 1, open_slots)
            return next_pos, open_slots

        fill(0, 0, 1)
        flat = []
        for token in tokens:
            flat.extend(token)
        return flat


def create_pop_for_deap(
    pop_size=10,
    n_vars=2,
    const_min=-1,
    const_max=5,
    method='random',
    fitness_function_for_deap_str=None,
    binary_operators=None,
    unary_operators=None,
    max_depth=35,
    max_tokens=None,
    max_attempts_per_individual=100,
    oblesa_population=None,
):
    """
    Create an initial population for symbolic regression.

    The representation is a fixed-length prefix vector with a configurable
    maximum depth. The default max_depth is 35, but most generated equations
    will be much shallower because projected leaf nodes terminate early.
    """
    var_list = [f'X{i}' for i in range(n_vars)]
    parser = SrVecParser(
        var_list=var_list,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        max_depth=max_depth,
        max_tokens=max_tokens,
    )
    bounds = parser.get_bounds_list(const_min=const_min, const_max=const_max)

    def to_outputs(individual):
        eq_string = parser.decode_equation(individual)
        deap_expr = parser.parse_srt_eq_to_deap_representation(eq_string)
        canonical_expr = parser.canonicalize_expression(deap_expr)
        return eq_string, deap_expr, canonical_expr

    def generate_random_unique_population():
        population = []
        seen = set()
        total_attempts = max(pop_size * max_attempts_per_individual, pop_size)

        attempts = 0
        while len(population) < pop_size and attempts < total_attempts:
            attempts += 1
            individual = parser.random_individual(const_min=const_min, const_max=const_max)
            _, _, canonical_expr = to_outputs(individual)
            key = canonical_expr
            if key in seen:
                continue
            seen.add(key)
            population.append(individual)

        if len(population) < pop_size:
            raise RuntimeError(
                f'Could not generate {pop_size} unique individuals after {attempts} attempts. '
                f'Generated {len(population)} unique individuals. '
                f'Try increasing max_attempts_per_individual, max_tokens, or the operator set.'
            )
        return population

    if method == 'random':
        population = generate_random_unique_population()

    elif method == 'oblesa':
        if fitness_function_for_deap_str is None:
            raise ValueError('fitness_function_for_deap_str must be provided for oblesa initialization.')

        def fitness_function(individual):
            _, deap_expr, _ = to_outputs(individual)
            return fitness_function_for_deap_str(deap_expr)

        from pyBlindOpt.init import oblesa
        import numpy as np

        if oblesa_population is not None:
            # Keep the original behavior when explicit seed equations are provided:
            # initialize OBLESA with the user-provided expressions plus the best
            # individuals from the balanced random population.
            encoded_population = (
                generate_random_unique_population()
            )

            n_fill = len(encoded_population) - len(oblesa_population)
            if n_fill > 0:
                scored_population = [
                    (fitness_function(individual), individual)
                    for individual in encoded_population
                ]
                encoded_population = [
                    ind for _, ind in heapq.nlargest(n_fill, scored_population, key=lambda x: x[0])
                ]
            else:
                encoded_population = []

            for expr in oblesa_population:
                try:
                    encoded = parser.encode_equation(expr)
                    encoded_population.append(encoded)
                except ValueError as exc:
                    raise ValueError(f"Error encoding OBLESA expression '{expr}': {exc}") from exc

            encoded_population = np.array(encoded_population)
        else:
            # New behavior: if no explicit OBLESA seed expressions are provided,
            # seed OBLESA with a balanced random population generated by this parser
            # instead of letting OBLESA start from scratch.
            encoded_population = (
                generate_random_unique_population()
            )
            encoded_population = np.array(encoded_population)

        print(f"The optimization space for oblesa has {len(bounds)} dimensions and the bounds are: {bounds[0:5]} ... {bounds[-5:]}")

        rng = np.random.default_rng(42)
        raw_population = oblesa(
            seed = rng,
            objective=fitness_function,
            bounds=np.array(bounds),
            n_pop=pop_size,
            population=encoded_population,
            opp='standard',
        )
        raw_population = [list(ind) for ind in raw_population]

        population = []
        seen = set()
        for individual in raw_population:
            _, _, canonical_expr = to_outputs(individual)
            if canonical_expr in seen:
                # print(f"OBLESA generated a duplicate expression: {canonical_expr}. Skipping.")
                continue
            seen.add(canonical_expr)
            population.append(individual)

        refill_attempts = 0
        total_refill_attempts = max((pop_size - len(population)) * max_attempts_per_individual, 0)
        while len(population) < pop_size and refill_attempts < total_refill_attempts:
            refill_attempts += 1
            individual = parser.random_individual(const_min=const_min, const_max=const_max)
            _, _, canonical_expr = to_outputs(individual)
            if canonical_expr in seen:
                continue
            seen.add(canonical_expr)
            population.append(individual)

        if len(population) < pop_size:
            raise RuntimeError(
                f'OBLESA returned duplicates and the refill stage could not complete {pop_size} unique individuals. '
                f'Generated {len(population)} unique individuals. '
                f'Try increasing max_attempts_per_individual, max_tokens, or the operator set.'
            )
    else:
        raise ValueError("method must be either 'random' or 'oblesa'.")

    print(f"Generated initial population of {len(population)} unique individuals using method '{method}'.")

    eq_strings = []
    deap_population = []
    for individual in population:
        eq_string, deap_expr, _ = to_outputs(individual)
        eq_strings.append(eq_string)
        deap_population.append(deap_expr)
    return eq_strings, deap_population


if __name__ == '__main__':
    eqs, deap_eqs = create_pop_for_deap(
        pop_size=10,
        n_vars=19,
        unary_operators=['sqrt', 'exp', 'log', 'sq2', 'sq3'],
        max_depth=50,
    )
    for i, expr in enumerate(eqs, start=1):
        print(f'{i}: {expr}')

    parser = SrVecParser(
        var_list=[f"X{i}" for i in range(19)],
        unary_operators=["sqrt", "exp", "log", "sq2", "sq3"],
        binary_operators=["add", "sub", "mul", "div"],
        max_depth=50,
        max_tokens=5000,
    )

    your_equation_string = "sqrt(sqrt(div(sqrt(sqrt(div(sqrt(exp(sqrt(sqrt(mul(mul(sqrt(mul(mul(div(X5, X11), div(sqrt(add(add(X8, X2), 98.38333678666703)), add(sqrt(X10), exp(X13)))), add(add(sub(div(X15, div(exp(div(X4, add(X1, X16))), X10)), X5), div(div(sq2(-50.12980327610712), X8), sqrt(X2))), sqrt(sq3(X5))))), sq3(X0)), add(mul(mul(log(sq2(sq3(X11))), add(sqrt(exp(sub(mul(X0, X6), sqrt(add(sq2(sub(mul(sqrt(sq3(sqrt(X13))), X6), X11)), mul(exp(X5), X16)))))), X9)), sqrt(X11)), add(add(sub(div(mul(sqrt(mul(mul(exp(sqrt(X18)), div(sqrt(X2), add(sq3(sq2(sq3(X15))), X3))), add(sq3(X5), sq3(X5)))), X0), X6), X5), sqrt(sq3(div(div(add(mul(sq2(-24.46086420986792), X1), 98.38333678666703), add(sq2(sq2(sub(add(X7, X13), exp(div(sqrt(sq3(X5)), sqrt(X4)))))), exp(sq3(X0)))), add(sq2(sq2(sub(X5, exp(div(sqrt(add(div(mul(X0, X11), X2), sq3(X9))), log(39.57535868108786)))))), exp(sq3(X0))))))), sqrt(sq3(X5))))))))), add(add(div(exp(div(add(log(add(div(add(div(exp(div(add(add(sqrt(X6), X9), sqrt(mul(X6, X1))), exp(mul(add(div(sqrt(sqrt(X15)), log(sqrt(X6))), add(X9, X7)), add(sqrt(div(mul(X0, sqrt(sqrt(div(X18, X6)))), sqrt(X1))), div(add(add(X9, X12), X9), div(mul(X8, X4), add(X3, 98.38333678666703)))))))), div(exp(X12), mul(X1, add(mul(sq3(X14), sqrt(add(X7, X15))), X5)))), add(div(div(exp(div(add(div(sub(X10, div(X4, X2)), sqrt(X10)), sqrt(mul(X17, mul(add(sq3(X0), sq3(X0)), X1)))), exp(add(mul(div(exp(X6), X2), X7), X13)))), div(X11, X1)), exp(sub(sq3(sqrt(sq2(sub(sub(sq3(X17), X14), sqrt(X18))))), sqrt(log(mul(div(sq3(add(X8, mul(X16, X2))), sq3(X8)), add(X2, sq3(sq3(X1))))))))), div(X5, sqrt(X2)))), exp(sq2(X14))), div(add(X13, sqrt(X5)), add(sq2(X1), sq2(sq3(div(X1, div(X4, sqrt(X13))))))))), sqrt(sqrt(sq2(sub(log(X1), sqrt(exp(sq2(exp(X7))))))))), exp(div(X5, add(X11, X11))))), exp(X13)), div(add(X9, mul(add(add(div(X1, div(mul(add(sq3(exp(X0)), sq3(sqrt(div(sqrt(exp(sqrt(sqrt(mul(mul(sqrt(mul(mul(div(X5, X11), div(sqrt(add(add(X8, X2), 98.38333678666703)), add(sqrt(X10), exp(X13)))), add(add(sub(div(X15, div(exp(div(X4, add(X1, X16))), X10)), X5), div(div(sq2(-50.12980327610712), X8), sqrt(X2))), sqrt(sq3(X5))))), sq3(X0)), add(mul(mul(log(sq2(sq3(X11))), add(sqrt(exp(sub(mul(X0, X6), sqrt(add(sq2(sub(mul(sqrt(sq3(sqrt(X13))), X6), X11)), mul(exp(sqrt(log(exp(X18)))), X16)))))), X9)), sqrt(X11)), add(add(sub(div(mul(sqrt(mul(mul(exp(sqrt(X18)), div(sqrt(X2), add(sq3(sq2(sq3(X15))), X3))), add(sq3(X5), sq3(X5)))), X0), X6), X5), sqrt(sq3(div(div(add(mul(sq2(-24.46086420986792), X1), 98.38333678666703), add(sq2(sq2(sub(add(X7, X13), exp(div(sqrt(sq3(X5)), sqrt(X4)))))), exp(sq3(X0)))), add(sq2(sq2(sub(X5, exp(div(sqrt(add(div(mul(X0, X11), X2), sq3(X9))), log(39.57535868108786)))))), exp(sq3(X0))))))), sqrt(sq3(X5))))))))), add(add(div(exp(div(add(log(add(div(add(div(exp(div(add(add(sqrt(X6), X9), sqrt(mul(X6, X1))), exp(mul(add(div(sqrt(sqrt(X15)), log(sqrt(X6))), add(X9, X7)), add(sqrt(div(mul(X0, sqrt(sqrt(div(X18, X6)))), sqrt(X1))), div(add(add(X9, X12), X9), div(mul(X8, X4), add(X3, 98.38333678666703)))))))), div(exp(X12), mul(X1, add(mul(sq3(X14), sqrt(add(X7, X15))), X5)))), add(div(div(exp(div(add(div(sub(X10, div(X4, X2)), sqrt(X10)), sqrt(mul(X17, mul(add(sq3(X0), sq3(X0)), X1)))), exp(add(mul(div(exp(X6), X2), X7), X13)))), div(X11, X1)), exp(sub(sq3(sqrt(sq2(sub(sub(sq3(X17), X14), sqrt(X18))))), sqrt(log(mul(div(sq3(add(X8, mul(X16, X2))), sq3(X8)), add(X2, sq3(sq3(X1))))))))), div(X5, sqrt(X2)))), exp(sq2(X14))), div(add(X13, sqrt(X5)), add(sq2(X1), sq2(sq3(div(X1, div(X4, sqrt(X13))))))))), sqrt(sqrt(sq2(sub(log(X1), sqrt(exp(sq2(exp(X7))))))))), exp(div(X5, add(X11, X11))))), exp(X13)), div(add(X9, mul(add(add(div(X1, div(mul(add(sq3(exp(X0)), sq3(sqrt(X3))), exp(div(X10, X6))), sq2(add(-97.92877186254793, X11)))), X5), sq3(div(sq3(X15), div(X6, log(sqrt(X2)))))), exp(sub(log(X11), sqrt(sq3(div(X4, X2))))))), X8)), sq3(div(exp(X15), exp(X1)))))))), exp(div(X10, X6))), sq2(add(-97.92877186254793, X11)))), X5), sq3(div(sq3(X15), div(X6, log(sqrt(X2)))))), exp(sub(log(X11), sqrt(sq3(div(X4, X2))))))), X8)), sq3(div(exp(X15), exp(X1))))))), X1)))"

    encoded = parser.encode_equation(your_equation_string)
    decoded = parser.decode_equation(encoded)

    print(decoded)
