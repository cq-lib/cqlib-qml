"""Parse constructor-like configuration without evaluating Python code."""
import ast
import inspect


def parse_configuration(value, registry, nested=None):
    """Accept a registered name or keyword-only call with literal values.

    Nested calls are restricted to the supplied scheduler registry. Attribute
    access, positional arguments and expanded arguments are never evaluated.
    """
    try:
        tree = ast.parse(value.strip(), mode='eval').body
    except (SyntaxError, TypeError, AttributeError) as exc:
        raise ValueError(f'Invalid configuration: {value!r}') from exc

    def construct(node, constructors):
        if isinstance(node, ast.Name):
            name, keywords = node.id, []
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.args:
            name, keywords = node.func.id, node.keywords
        else:
            raise ValueError('Configuration must be a registered name or keyword-only call')
        constructor = constructors.get(name.casefold())
        if constructor is None:
            raise ValueError(f'Unsupported configuration name: {name}')
        allowed = {key for key, parameter in inspect.signature(constructor).parameters.items()
                   if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)}
        kwargs = {}
        for keyword in keywords:
            key = keyword.arg
            if key not in allowed or key in kwargs:
                raise ValueError(f'Unknown or duplicate parameter: {key}')
            if isinstance(keyword.value, ast.Call) and key == 'lr_scheduler' and nested:
                kwargs[key] = construct(keyword.value, nested)
            else:
                try:
                    kwargs[key] = ast.literal_eval(keyword.value)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'Parameter {key} must be a literal') from exc
        return constructor(**kwargs)

    return construct(tree, registry)


def same_parameter_value(old, new):
    """Compare scalar configurations without assuming array equality is scalar."""
    if old is new:
        return True
    try:
        return bool(old == new)
    except (TypeError, ValueError):
        return False
