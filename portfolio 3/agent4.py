import numpy as np


class Agent4:
    """Random baseline agent."""

    def act(self, observation) -> int:
        """Return random action."""
        return int(np.random.randint(6))