class SetupError(Exception):
    """Exception raised when the environment is not setup correctly."""
    def __init__(self, package, env):
        self.message = (
            f'Failed to create environment: {env}. '
            f'pip install with [{package}] to resolve this issue. '
            'Some environments require non-python dependencies. '
            'These are included in PufferTank.'
        )
        super().__init__(self.message)

class EnvironmentSetupError(RuntimeError):
    def __init__(self, message="Environment setup error."):
        self.message = message
        super().__init__(self.message)

class APIUsageError(RuntimeError):
    """Exception raised when the API is used incorrectly."""

    def __init__(self, message="API usage error."):
        self.message = message
        super().__init__(self.message)

class InvalidAgentError(ValueError):
    """Exception raised when an invalid agent key is used."""

    def __init__(self, agent_id, agents):
        message = (
            f'Invalid agent/team ({agent_id}) specified. '
            f'Valid values:\n{agents}'
        )
        super().__init__(message)