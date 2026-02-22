"""
ACD-KEL Custom Exception Hierarchy
Provides structured error handling for cross-domain operations
"""

class ACDKELException(Exception):
    """Base exception for all ACD-KEL operations"""
    def __init__(self, message: str, domain: str = "global"):
        self.message = message
        self.domain = domain
        super().__init__(f"[{domain}] {message}")

class CompatibilityError(ACDKELException):
    """Raised when domain knowledge cannot be harmonized"""
    def __init__(self, source_domain: str, target_domain: str, data_type: str):
        message = f"Cannot harmonize {data_type} from {source_domain} to {target_domain}"
        super().__init__(message, source_domain)
        self.target_domain = target_domain
        self.data_type = data_type

class ValidationError(ACDKELException):
    """Raised when knowledge validation fails"""
    def __init__(self, domain: str, field: str, value: str, constraint: str):
        message = f"Validation failed for {field}={value} (constraint: {constraint})"
        super().__init__(message, domain)
        self.field = field
        self.constraint = constraint

class DomainRegistrationError(ACDKELException):
    """Raised when domain registration fails"""
    def __init__(self, domain: str, reason: str):
        message = f"Domain registration failed: {reason}"
        super().__init__(message, domain)

class KnowledgeSyncError(ACDKELException):
    """Raised when knowledge synchronization fails"""
    def __init__(self, operation: str, domain: str, error_details: str):
        message = f"Sync failed for {operation}: {error_details}"
        super().__init__(message, domain)
        self.operation = operation