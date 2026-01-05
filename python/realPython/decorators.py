import functools # Importing functools to use wraps. It helps to preserve the original function's metadata.

def do_twice(func):
    @functools.wraps(func) # This preserves the original function's metadata. however, it is optional.
    def wrapper_do_twice(*one_arg, **another_arg):
        func(*one_arg, **another_arg)
        return func(*one_arg, **another_arg)
    return wrapper_do_twice

def decorators(func):
    @functools.wraps(func) # This preserves the original function's metadata. however, it is optional.
    def wrapper_decorators(*args, **kwargs):
        # Do something before calling the function
        value = func(*args, **kwargs)
        # Do something after calling the function
        return value  # Ensure the original function's return value is returned
    return wrapper_decorators


import time

def timer(func):
    """Print the runtime of the decorator functions

    Args:
        func (_type_): _description_

    Returns:
        _type_: _description_
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

def debug(func):
    """ print the function isgnature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwrags_repr = [f"{k}={repr(v)}" for k,v in kwargs.items()]
        print(f"calling {func.__name__}({', '.join(args_repr + kwrags_repr)})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")
        return value
    return wrapper_debug


def slow_down(func):
    """Sleep 1 second before calling the function"""
    @functools.wraps(func)
    def wrapper_slow_down(*args, **kwargs):
        time.sleep(1)
        return func(*args, **kwargs)
    return wrapper_slow_down