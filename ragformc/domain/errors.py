class RagForMcError(Exception):
    """Base exception for the application."""


class ConfigurationError(RagForMcError):
    """Raised when required configuration is missing or invalid."""


class StartupValidationError(RagForMcError):
    """Raised when the vector database cannot satisfy runtime expectations."""


class ServiceError(RagForMcError):
    """Raised when an external service request fails."""
