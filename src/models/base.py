"""Common interface for all recommendation models."""

from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    """Base class shared by all recommendation models."""

    @abstractmethod
    def fit(self, train_data, valid_data=None):
        """Fit the model using training data."""
        raise NotImplementedError

    @abstractmethod
    def recommend(self, test_data, top_k=10):
        """Generate Top-K recommendations for test users."""
        raise NotImplementedError

    def save(self, path):
        """Save model parameters when necessary."""
        raise NotImplementedError

    def load(self, path):
        """Load saved model parameters when necessary."""
        raise NotImplementedError
