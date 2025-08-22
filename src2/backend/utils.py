from pathlib import Path
import sys,os

from pathlib import Path

def resource_path_prompts(filename: str) -> str:

    # resolve() -> absolute path of the file,here, utils.py...so D:\e_learning_bot\src2\backend\utils.py

    base = Path(__file__).resolve().parent  # -> backend (parent)
    return str(base  / "prompts" / filename)




def resource_path_2(filename: str) -> str:
    base = Path(__file__).resolve().parents[1]  # -> src2 (parent.parent)->grandparent
    return str(base  / filename)


def resource_path(relative_path):
    """
    Resolve a relative path to absolute, compatible with PyInstaller and dev.
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)




import traceback
from functools import wraps

def safe_run(default_return=None, log_traceback=True):
    """
    Decorator to catch exceptions and return a default response.

    Args:
        default_return: What to return in case of exception.
        log_traceback: Whether to print the full traceback.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_traceback:
                    print(f"Error in {func.__name__}:")
                    traceback.print_exc()
                # Return a structured error dictionary by default
                return {
                    "status": "error",
                    "message": str(e)
                } if default_return is None else default_return
        return wrapper
    return decorator


