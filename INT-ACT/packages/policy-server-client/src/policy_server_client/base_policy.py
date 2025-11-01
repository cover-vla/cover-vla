import abc


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: dict) -> dict:
        """Infer actions from observations."""
        raise NotImplementedError("infer() not implemented")

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the policy to its initial state."""
        raise NotImplementedError("reset() not implemented")
