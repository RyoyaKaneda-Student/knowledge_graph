def min_or_default(default, vs_min: int = None):
    return min(default, vs_min) if vs_min is not None else default
