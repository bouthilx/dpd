def compare_dict(d1, d2, depth=1):
    indent = ' ' * depth * 2
    print(f'{indent}{{')

    for k, v in d1.items():
        v2 = d2.get(k)
        indent2 = ' ' * (depth + 1) * 2

        if isinstance(v, dict) and isinstance(v2, dict):
            print(f'{indent2}{k}: ', end='')
            compare_dict(v, v2, depth + 1)
        elif isinstance(v, list) and isinstance(v2, list):
            print(f'{indent2}{k}: [', end='')
            for d11, d22 in zip(v, v2):
                compare_dict(d11, d22, depth + 1)
        else:
            print(f'{indent2}{k}: {v} => {v2}')

    print(f'{indent}}}')
