from __future__ import annotations

from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    TypeVar,
)

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

if TYPE_CHECKING:
    from replicate.client import Client
    from replicate.prediction import Prediction

_T = TypeVar("_T")


def _identity(x: _T) -> _T:
    """Identity function which returns its input"""
    return x


class ReplicateEmbeddings(BaseModel, Embeddings):
    """Replicate embedding models.

    To use, you should have the ``replicate`` python package installed,
    and the environment variable ``REPLICATE_API_TOKEN`` set with your API token.
    You can find your token here: https://replicate.com/account

    The model param is required, but any other model parameters can also
    be passed in with the format model_kwargs={model_param: value, ...}

    Example:
        .. code-block:: python

            from langchain_community.embeddings import ReplicateEmbeddings

            replicate = ReplicateEmbeddings(
                model="ibm-granite/granite-embedding-278m-multilingual",
            )
    """

    model: str
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    replicate_api_token: Optional[str] = None
    texts_key: Optional[str] = None
    texts_value_mapping: Callable[[List[str]], Any] = Field(
        default=_identity,
        exclude=True,
        repr=False,
    )
    """Can be used to map the input list of strings for the embeddings to some
        type other than List[str]. For example, if the model requires the strings as a JSON
        formatted string, this field can be set to `json.dumps`.
    """
    version_obj: Any = Field(default=None, exclude=True)
    """Optionally pass in the model version object during initialization to avoid
        having to make an extra API call to retrieve it during streaming. NOTE: not
        serializable, is excluded from serialization.
    """
    _replicate_client: Optional[Client] = PrivateAttr(default=None)

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    @property
    def lc_secrets(self) -> Dict[str, str]:  # pylint: disable=missing-function-docstring
        return {"replicate_api_token": "REPLICATE_API_TOKEN"}

    def _create_prediction(self, texts: List[str], **kwargs: Any) -> Prediction:
        try:
            import replicate as replicate_python  # pylint: disable=import-outside-toplevel
        except ImportError:
            raise ImportError(  # pylint: disable=raise-missing-from
                "Could not import replicate python package. "
                "Please install it with `pip install replicate`."
            )

        # get the replicate client
        if self._replicate_client is None:
            self._replicate_client = (
                replicate_python.Client(api_token=self.replicate_api_token)
                if self.replicate_api_token
                else replicate_python.default_client
            )
        # get the model and version
        if self.version_obj is None:
            if ":" in self.model:
                model_str, version_str = self.model.split(":")
                model = self._replicate_client.models.get(model_str)
                self.version_obj = model.versions.get(version_str)
            else:
                model = self._replicate_client.models.get(self.model)
                self.version_obj = model.latest_version

        if self.texts_key is None:
            # sort through the openapi schema to get the name of the first input
            input_properties = sorted(
                self.version_obj.openapi_schema["components"]["schemas"]["Input"][  # type: ignore
                    "properties"
                ].items(),
                key=lambda item: item[1].get("x-order", 0),
            )
            self.texts_key = input_properties[0][0]

        input_: Dict = {
            self.texts_key: self.texts_value_mapping(texts),
            **self.model_kwargs,
            **kwargs,
        }

        # if it's an official model
        if ":" not in self.model:
            return self._replicate_client.models.predictions.create(
                self.model, input=input_
            )

        return self._replicate_client.predictions.create(
            version=self.version_obj,
            input=input_,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a Replicate embeddings model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        prediction = self._create_prediction(texts)
        prediction.wait()
        if prediction.status == "failed":
            raise RuntimeError(prediction.error)
        completion = prediction.output
        assert isinstance(completion, list)
        return completion

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a Replicate embeddings model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
