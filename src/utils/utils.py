import gc


def force_gc():
    gc.collect()


# decorator
def force_gc_after_function(func):
    def wrapper(*args, **kwargs):
        rev = func(*args, **kwargs)
        force_gc()
        return rev

    return wrapper
