"""
Utility functions for Freshdesk KB Chatbot Backend.

Includes LangChain's RecursiveCharacterTextSplitter for text chunking.
"""

import logging
import re
from typing import List, Optional, Any
from functools import wraps

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up logging
logger = logging.getLogger(__name__)


# LangChain text splitter - replaces manual tiktoken chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def chunk_text(text: str, max_tokens: int = 512, overlap_tokens: int = 50) -> List[str]:
    """
    Chunk text into smaller pieces using LangChain's RecursiveCharacterTextSplitter.

    Uses smart chunking that respects sentence boundaries and document structure.

    Args:
        text: Text to chunk
        max_tokens: Maximum characters per chunk (default: 512)
        overlap_tokens: Number of characters to overlap between chunks (default: 50)

    Returns:
        List of text chunks

    Example:
        >>> text = "This is a long article..." * 100
        >>> chunks = chunk_text(text, max_tokens=512)
        >>> len(chunks)
        5
    """
    if not text or not text.strip():
        return []

    # Use LangChain's splitter with custom chunk size if needed
    if max_tokens != 512 or overlap_tokens != 50:
        custom_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=overlap_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        return custom_splitter.split_text(text)
    
    return text_splitter.split_text(text)


def chunk_documents(documents: List[Any], max_tokens: int = 512, overlap_tokens: int = 50) -> List[Any]:
    """
    Chunk LangChain Document objects using RecursiveCharacterTextSplitter.
    
    Preserves metadata across chunks.
    
    Args:
        documents: List of LangChain Document objects to chunk.
        max_tokens: Maximum characters per chunk.
        overlap_tokens: Number of characters to overlap.
        
    Returns:
        List of chunked Document objects.
    """
    if not documents:
        return []
    
    if max_tokens != 512 or overlap_tokens != 50:
        custom_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=overlap_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        return custom_splitter.split_documents(documents)
    
    return text_splitter.split_documents(documents)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> callable:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorated function

    Example:
        >>> @retry_with_backoff(max_retries=3, initial_delay=1.0)
        ... def fetch_data():
        ...     # May fail temporarily
        ...     return api.get_data()
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        import time
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )

            raise last_exception

        return wrapper
    return decorator


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to maximum length, adding suffix if truncated.

    Args:
        text: Text to truncate
        max_length: Maximum length (including suffix)
        suffix: Suffix to add if truncated

    Returns:
        Truncated text

    Example:
        >>> truncate_text("This is a very long text", max_length=10)
        'This is...'
    """
    if not text:
        return ""

    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def sanitize_text(text: str) -> str:
    """
    Sanitize text by removing excessive whitespace and control characters.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)

    # Remove control characters except newlines and tabs
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')

    return text.strip()


def format_article_url(domain: str, article_id: int) -> str:
    """
    Format a Freshdesk article URL.

    Args:
        domain: Freshdesk domain (subdomain)
        article_id: Article ID

    Returns:
        Full article URL

    Example:
        >>> format_article_url("mycompany", 12345)
        'https://mycompany.freshdesk.com/support/solutions/articles/12345'
    """
    return f"https://{domain}.freshdesk.com/support/solutions/articles/{article_id}"


def extract_article_id_from_url(url: str) -> Optional[int]:
    """
    Extract article ID from Freshdesk URL.

    Args:
        url: Freshdesk article URL

    Returns:
        Article ID or None if not found

    Example:
        >>> extract_article_id_from_url("https://mycompany.freshdesk.com/support/solutions/articles/12345")
        12345
    """
    match = re.search(r'/articles/(\d+)', url)
    if match:
        return int(match.group(1))
    return None


def batch_items(items: List, batch_size: int) -> List[List]:
    """
    Batch items into smaller lists.

    Args:
        items: List of items to batch
        batch_size: Size of each batch

    Returns:
        List of batches

    Example:
        >>> batch_items([1, 2, 3, 4, 5], batch_size=2)
        [[1, 2], [3, 4], [5]]
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])

    return batches


def safe_get(dictionary: dict, *keys: str, default: Any = None) -> Any:
    """
    Safely get nested dictionary values.

    Args:
        dictionary: Dictionary to get value from
        *keys: Sequence of keys to traverse
        default: Default value if key not found

    Returns:
        Value or default

    Example:
        >>> data = {"user": {"profile": {"name": "John"}}}
        >>> safe_get(data, "user", "profile", "name")
        'John'
        >>> safe_get(data, "user", "settings", "theme", default="dark")
        'dark'
    """
    current = dictionary
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


# Logging helper
def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    import logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
