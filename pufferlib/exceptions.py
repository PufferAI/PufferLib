class APIUsageError(RuntimeError):
    """Exception raised when the API is used incorrectly."""

    def __init__(self, message="API usage error."):
        self.message = message
        super().__init__(self.message)

class InvalidAgentError(ValueError):
    """Exception raised when an invalid agent key is used."""

    def __init__(self, team_id, teams):
        message = (
            f'Invalid agent/team ({team_id}) specified. '
            f'Valid teams:\n{list(teams.values())}'
        )
        super().__init__(message)

