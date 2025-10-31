from fastapi import HTTPException, status

class DetailedHTTPException(HTTPException):
    """
    A base custom HTTP exception class that allows for more detailed error messages.
    """
    def __init__(self, status_code: int, detail: str = "An unexpected error occurred", headers: dict = None):
        super().__init__(status_code=status_code, detail=detail, headers=headers)

class ModelLoadingError(DetailedHTTPException):
    """
    Exception raised when ML models or artifacts fail to load.
    """
    def __init__(self, detail: str = "Machine learning models failed to load. Please check server logs.", headers: dict = None):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail, headers=headers)

class PredictionProcessingError(DetailedHTTPException):
    """
    Exception raised when there's an error during the prediction pipeline.
    """
    def __init__(self, detail: str = "Error processing prediction request.", headers: dict = None):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail, headers=headers)

class DatabaseError(DetailedHTTPException):
    """
    Exception raised for database-related issues.
    """
    def __init__(self, detail: str = "A database operation failed.", headers: dict = None):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail, headers=headers)

class InvalidInputError(DetailedHTTPException):
    """
    Exception raised when input data is invalid.
    """
    def __init__(self, detail: str = "Invalid input provided.", headers: dict = None):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail, headers=headers)

