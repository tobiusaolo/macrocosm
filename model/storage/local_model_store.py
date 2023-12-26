import abc
from model.data import Model, ModelId


class LocalModelStore(abc.ABC):
    """An abstract base class for storing and retrieving a pre trained model locally."""

    @abc.abstractmethod
    def store_model(self, hotkey: str, model: Model) -> ModelId:
        """Stores a trained model in the appropriate location based on implementation."""
        pass

    @abc.abstractmethod
    def get_path(self, hotkey: str, model_id: ModelId) -> str:
        """Returns the path to the appropriate location based on implementation."""
        pass

    @abc.abstractmethod
    def retrieve_model(self, hotkey: str, model_id: ModelId) -> Model:
        """Retrieves a trained model from the appropriate location based on implementation."""
        pass

    @abc.abstractmethod
    def delete_models(self, hotkey: str):
        """Deletes all models for a given hotkey."""
        pass

    @abc.abstractmethod
    def delete_model(self, hotkey: str, model_id: ModelId):
        """Delete the given model."""
        pass
