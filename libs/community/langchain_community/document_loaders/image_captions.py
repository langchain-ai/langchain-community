from io import BytesIO
from pathlib import Path
from typing import Any, List, Tuple, Union, Optional

import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from copy import deepcopy


class ImageCaptionLoader(BaseLoader):
    """Load image captions.

    By default, the loader utilizes the pre-trained
    Salesforce BLIP image captioning model.
    https://huggingface.co/Salesforce/blip-image-captioning-base
    """

    def __init__(
        self,
        images: Union[str, Path, bytes, List[Union[str, bytes, Path]]],
        blip_processor: str = "Salesforce/blip-image-captioning-base",
        blip_model: str = "Salesforce/blip-image-captioning-base",
        decode_args: Optional[list[Any]] = None,
        decode_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Initialize with a list of image data (bytes) or file paths

        Args:
            images: Either a single image or a list of images. Accepts
                    image data (bytes) or file paths to images.
            blip_processor: The name of the pre-trained BLIP processor.
            blip_model: The name of the pre-trained BLIP model.
        """
        if isinstance(images, (str, Path, bytes)):
            self.images = [images]
        else:
            self.images = images

        self.blip_processor = blip_processor
        self.blip_model = blip_model
        # deepcopy to prevent modifying from outside,
        # not sure should I use it or not considering probably few would modify those
        if decode_args is None:
            decode_args = []
        self.decode_args = deepcopy(decode_args)

        if decode_kwargs is None:
            decode_kwargs = {}
        self.decode_kwargs = deepcopy(decode_kwargs)

    def load(self) -> List[Document]:
        """Load from a list of image data or file paths"""
        general_blip_processor = None
        general_blip_for_conditional_generation = None
        try:
            if "blip2" in self.blip_processor:
                from transformers import Blip2Processor

                general_blip_processor = Blip2Processor
            else:
                from transformers import BlipProcessor

                general_blip_processor = BlipProcessor

            if "blip2" in self.blip_model:
                from transformers import Blip2ForConditionalGeneration

                general_blip_for_conditional_generation = Blip2ForConditionalGeneration
            else:
                from transformers import BlipForConditionalGeneration

                general_blip_for_conditional_generation = BlipForConditionalGeneration
        except ImportError:
            raise ImportError(
                "`transformers` package not found, please install with "
                "`pip install transformers`."
            )

        processor = general_blip_processor.from_pretrained(self.blip_processor)
        model = general_blip_for_conditional_generation.from_pretrained(self.blip_model)

        results = []
        for image in self.images:
            caption, metadata = self._get_captions_and_metadata(
                model=model, processor=processor, image=image
            )
            doc = Document(page_content=caption, metadata=metadata)
            results.append(doc)

        return results

    def _get_captions_and_metadata(
        self, model: Any, processor: Any, image: Union[str, Path, bytes]
    ) -> Tuple[str, dict]:
        """Helper function for getting the captions and metadata of an image."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "`PIL` package not found, please install with `pip install pillow`"
            )

        image_source = image  # Save the original source for later reference

        try:
            if isinstance(image, bytes):
                image = Image.open(BytesIO(image)).convert("RGB")
            elif isinstance(image, str) and (
                image.startswith("http://") or image.startswith("https://")
            ):
                image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
            else:
                image = Image.open(image).convert("RGB")
        except Exception:
            if isinstance(image_source, bytes):
                msg = "Could not get image data from bytes"
            else:
                msg = f"Could not get image data for {image_source}"
            raise ValueError(msg)

        inputs = processor(image, "an image of", return_tensors="pt")
        output = model.generate(**inputs)

        caption: str = processor.decode(
            output[0], *self.decode_args, **self.decode_kwargs
        )
        if isinstance(image_source, bytes):
            metadata: dict = {"image_source": "Image bytes provided"}
        else:
            metadata = {"image_path": str(image_source)}

        return caption, metadata
