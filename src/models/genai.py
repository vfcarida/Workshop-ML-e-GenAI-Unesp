"""
GenAI module for Multimodal and Image Generation models.

Provides simplified wrappers around Vertex AI Generative Models (Gemini)
and Hugging Face Diffusers (Stable Diffusion) for educational purposes.
"""

import logging
from typing import List, Union, Any, Generator

try:
    from vertexai.preview.generative_models import GenerativeModel, Content, Image, GenerationResponse
except ImportError:
    logging.warning("vertexai not installed. Gemini functions will not work.")

try:
    import torch
    from diffusers import StableDiffusionPipeline
except ImportError:
    logging.warning("diffusers or torch not installed. Stable Diffusion will not work.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiWrapper:
    """
    Wrapper for Google Cloud Vertex AI Gemini Models.
    """
    
    def __init__(self, model_name: str = "gemini-pro-vision"):
        """
        Initializes the Gemini model.

        Args:
            model_name (str): The name of the Vertex AI model to use.
        """
        self.model_name = model_name
        try:
            self.model = GenerativeModel(model_name)
            logger.info(f"Successfully initialized {model_name}.")
        except NameError:
            logger.error("Vertex AI SDK is not available.")
            self.model = None
        except Exception as e:
            logger.error(f"Error initializing Vertex AI model: {e}")
            self.model = None

    def generate_content_stream(self, contents: List[Any]) -> Generator[str, None, None]:
        """
        Generates content from the model, yielding text chunks as they arrive.

        Args:
            contents (List[Any]): A list containing text prompts and/or Image objects.

        Yields:
            str: Generated text chunk.
        """
        if not self.model:
            logger.error("Model is not initialized.")
            return

        try:
            responses = self.model.generate_content(contents, stream=True)
            for response in responses:
                yield response.text
        except Exception as e:
            logger.error(f"Error during content generation: {e}")


class StableDiffusionWrapper:
    """
    Wrapper for Hugging Face Stable Diffusion.
    """
    
    def __init__(self, model_id: str = "CompVis/stable-diffusion-v1-4"):
        """
        Initializes the Stable Diffusion Pipeline.

        Args:
            model_id (str): The Hugging Face model ID.
        """
        self.model_id = model_id
        self.pipe = None
        try:
            # Check if CUDA is available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            logger.info(f"Loading Stable Diffusion on {self.device}...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id, 
                torch_dtype=dtype
            )
            self.pipe = self.pipe.to(self.device)
            logger.info("Stable Diffusion loaded successfully.")
        except NameError:
             logger.error("PyTorch or Diffusers is not installed.")
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion: {e}")

    def generate_images(self, prompt: str, num_images: int = 1) -> List[Any]:
        """
        Generates images from a text prompt.

        Args:
            prompt (str): Text description of the desired image.
            num_images (int, optional): Number of images to generate. Defaults to 1.

        Returns:
            List[PIL.Image.Image]: List of generated PIL Images.
        """
        if not self.pipe:
            logger.error("Stable Diffusion pipeline is not available.")
            return []
            
        try:
            prompts = [prompt] * num_images
            logger.info(f"Generating {num_images} images...")
            result = self.pipe(prompts)
            return result.images
        except Exception as e:
            logger.error(f"Error generating images: {e}")
            return []
