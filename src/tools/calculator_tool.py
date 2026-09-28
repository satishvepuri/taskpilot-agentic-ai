"""
Calculator tool — for numeric reasoning steps (e.g. "what's the refund
amount after a 15% restocking fee on $240?").

Deliberately avoids raw eval() on arbitrary LLM-generated strings (a real
security foot-gun in agent frameworks). Uses Python's `ast` module to parse
and validate the expression only contains safe arithmetic operations before
evaluating it.
"""
import ast
import operator

# Only these operations are permitted — no attribute access, no function
# calls, no imports. Anything else raises before it's ever executed.
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} is not allowed")
        return ALLOWED_OPERATORS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} is not allowed")
        return ALLOWED_OPERATORS[op_type](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculate(tool_input: str) -> dict:
    """tool_input: a plain arithmetic expression, e.g. '240 * (1 - 0.15)'"""
    try:
        parsed = ast.parse(tool_input, mode="eval")
        result = _safe_eval(parsed.body)
        return {"expression": tool_input, "result": result}
    except Exception as e:
        raise ValueError(f"Could not safely evaluate '{tool_input}': {e}")
