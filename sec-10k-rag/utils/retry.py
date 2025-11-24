
"""
Retry utilities with exponential backoff.
"""
import time
import logging
from functools import wraps
from typing import Callable, Tuple, Type, Any

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger_instance: logging.Logger = None
) -> Callable:
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each attempt
        exceptions: Tuple of exceptions to catch and retry
        logger_instance: Optional logger for retry messages
        
    Returns:
        Decorated function
        
    Example:
        @retry_with_backoff(max_attempts=3, exceptions=(requests.RequestException,))
        def fetch_data():
            return requests.get('https://api.example.com/data')
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            current_delay = initial_delay
            log = logger_instance or logger
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    
                    if attempt >= max_attempts:
                        log.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    log.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
            return None  # Should never reach here
            
        return wrapper
    return decorator


class RetryContext:
    """Context manager for retry logic."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.delay = delay
        self.exceptions = exceptions
        self.attempt = 0
        
    def __enter__(self):
        self.attempt += 1
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and issubclass(exc_type, self.exceptions):
            if self.attempt < self.max_attempts:
                time.sleep(self.delay)
                return True  # Suppress exception, retry
        return False  # Propagate exception