"""
Utility module for image visualization and processing.

This module provides functions to download, display, and format images,
commonly used for Generative AI and Multimodal ML tasks.
"""

import io
import logging
import typing
import urllib.request
import http.client

try:
    import IPython.display
except ImportError:
    logging.warning("IPython not available. Display features might not work outside Jupyter.")

from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps
from vertexai.preview.generative_models import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def display_image_compressed(pil_image: PIL_Image.Image) -> None:
    """
    Displays a compressed JPEG version of a PIL Image in an IPython environment.

    Args:
        pil_image (PIL.Image.Image): The PIL image to be displayed.
    """
    try:
        image_io = io.BytesIO()
        pil_image.save(image_io, "jpeg", quality=80, optimize=True)
        image_bytes = image_io.getvalue()
        ipython_image = IPython.display.Image(image_bytes)
        IPython.display.display(ipython_image)
    except NameError:
        logger.error("IPython.display is not available. Cannot display image.")
    except Exception as e:
        logger.error(f"Error displaying compressed image: {e}")

def display_image(image: Image, max_width: int = 600, max_height: int = 350) -> None:
    """
    Displays a Vertex AI Image object within a Jupyter notebook.

    Args:
        image (vertexai.preview.generative_models.Image): The Vertex AI Image object.
        max_width (int, optional): Maximum width for display. Defaults to 600.
        max_height (int, optional): Maximum height for display. Defaults to 350.
    """
    try:
        pil_image = typing.cast(PIL_Image.Image, getattr(image, '_pil_image', None))
        
        if pil_image is None:
             logger.error("The provided image does not contain a valid PIL image reference.")
             return

        if pil_image.mode != "RGB":
            # Modes such as RGBA are not yet supported by all Jupyter environments
            pil_image = pil_image.convert("RGB")
            
        image_width, image_height = pil_image.size
        if max_width < image_width or max_height < image_height:
            # Resize to display a smaller notebook image
            pil_image = PIL_ImageOps.contain(pil_image, (max_width, max_height))
            
        display_image_compressed(pil_image)
    except Exception as e:
        logger.error(f"Failed to display image: {e}")


def get_image_bytes_from_url(image_url: str) -> bytes:
    """
    Downloads an image from a given URL and returns its bytes.

    Args:
        image_url (str): The URL of the image to download.

    Returns:
        bytes: The image bytes.
        
    Raises:
        ValueError: If the downloaded file is not a valid image format (PNG/JPEG).
        urllib.error.URLError: If the URL cannot be fetched.
    """
    try:
        req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            response_typed = typing.cast(http.client.HTTPResponse, response)
            content_type = response_typed.headers.get("Content-Type", "")
            
            if content_type not in ("image/png", "image/jpeg"):
                logger.warning(f"Unexpected Content-Type: {content_type}. Proceeding anyway, but errors may occur.")
                
            image_bytes = response_typed.read()
            return image_bytes
    except Exception as e:
        logger.error(f"Failed to fetch image from {image_url}: {e}")
        raise

def load_image_from_url(image_url: str) -> Image:
    """
    Downloads an image from a URL and loads it into a Vertex AI Image object.

    Args:
        image_url (str): The URL of the image.

    Returns:
        vertexai.preview.generative_models.Image: The loaded Image object.
    """
    image_bytes = get_image_bytes_from_url(image_url)
    return Image.from_bytes(image_bytes)

def print_multimodal_prompt(contents: typing.List[typing.Any]) -> None:
    """
    Outputs a full multimodal prompt (text and images) for readability.
    
    Given a list of contents that would be sent to a multimodal model like Gemini,
    this function prints the text and displays the images.

    Args:
        contents (list): List of strings and/or Vertex AI Image objects.
    """
    for content in contents:
        if isinstance(content, Image):
            display_image(content)
        else:
            print(content)

def image_grid(imgs: typing.List[PIL_Image.Image], rows: int, cols: int) -> PIL_Image.Image:
    """
    Creates a grid of images.

    Args:
        imgs (List[PIL_Image.Image]): List of PIL images to place in the grid.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.

    Returns:
        PIL_Image.Image: A single PIL Image containing the grid.
        
    Raises:
        AssertionError: If the number of images doesn't match rows * cols.
    """
    assert len(imgs) == rows * cols, "Number of images must equal rows * cols"
    
    if not imgs:
        raise ValueError("The images list cannot be empty.")

    w, h = imgs[0].size
    grid = PIL_Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
        
    return grid
