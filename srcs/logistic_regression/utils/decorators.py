from json import JSONDecodeError
from functools import wraps

# List of exceptions to handle
KNOWN_EXCEPTIONS = [
    ValueError,
    KeyError,
    TypeError,
    PermissionError,
    FileNotFoundError,
    RuntimeError,
    AttributeError,
    JSONDecodeError,
    ImportError,
]


def error_handler(handle_exceptions=(), suppress: bool = False) -> callable:
    """
    Decorator to handle exceptions in the decorated method.

    Args:
        handle_exceptions (tuple): A tuple of exception
            types to handle.
        suppress (bool): If True, suppress the exception
            after logging it.

    Returns:
        function: The decorated method.
    """
    def decorator(method: callable) -> callable:
        """
        Decorator to handle exceptions in the decorated method.

        Args:
            method (function): The method to decorate.

        Returns:
            function: The decorated method.
        """
        @wraps(method)
        def wrapper(*args, **kwargs) -> any:
            """
            Wrapper function to handle exceptions.

            Args:
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.

            Raises:
                Exception: If suppress is False.

            Returns:
                Any: The result of the decorated method.
            """
            try:
                return method(*args, **kwargs)
            except handle_exceptions as e:
                if type(e) in KNOWN_EXCEPTIONS:
                    e_message = f"{type(e).__name__} in {method.__name__}: {e}"
                else:
                    e_message = f"Error in {method.__name__}: {e}"

                print(e_message)
                if not suppress:
                    raise
            except Exception as e:
                # Catch any other unexpected exceptions
                e_message = f"Unexpected error in {method.__name__}: {e}"
                print(e_message)
                raise

        return wrapper

    return decorator
