import time
import asyncio
from functools import wraps
from typing import Optional, Callable


def timeit(label: Optional[str] = None, *, show: bool = True) -> Callable:
    """
    decorator to time a function or method and print runtime in a readable format.

    Args:
        label: Optional label to identify the timed operation in output
        show: Whether to print the timing information (default: True)

    Returns:
        Decorated function that times its execution

    Example:
        Elapsed time [func name]: 0m 2.34s
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                result = await func(*args, **kwargs)
                elapsed = time.time() - start
                mins = int(elapsed // 60)
                secs = elapsed % 60
                lbl = f" [{label}]" if label else ""
                if show:
                    print(f"Elapsed time{lbl}: {mins}m {secs:.2f}s")
                return result
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                mins = int(elapsed // 60)
                secs = elapsed % 60
                lbl = f" [{label}]" if label else ""
                if show:
                    print(f"Elapsed time{lbl}: {mins}m {secs:.2f}s")
                return result
            return sync_wrapper
    return decorator

