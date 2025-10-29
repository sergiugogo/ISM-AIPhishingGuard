"""Custom exceptions for PhishGuard."""


class PhishGuardException(Exception):
    """Base exception for PhishGuard."""
    pass


class ModelLoadError(PhishGuardException):
    """Raised when model fails to load."""
    pass


class ValidationError(PhishGuardException):
    """Raised when input validation fails."""
    pass


class PredictionError(PhishGuardException):
    """Raised when prediction fails."""
    pass


class RateLimitExceeded(PhishGuardException):
    """Raised when rate limit is exceeded."""
    pass
