"""Function summarization.

The FunctionSummarizer class implements the CodeT5 model for function summarization.

Typical usage:

    docstring_node = FunctionSummarizer.create_docstring_node(function_node)
"""
# Base imports
from logging import getLogger

# pip imports
from transformers import RobertaTokenizer, T5ForConditionalGeneration

# Model imports
from repograph.entities.graph.models.nodes import Function

# Utils imports
from repograph.entities.summarization.utils import clean_source_code


# Setup logging
log = getLogger("repograph.entities.summarization.service")

import torch

class SummarizationService:
    tokenizer: any = None
    model: any = None
    active: bool

    def __init__(self, summarize: bool = False):
        """Constructor

        Args:
            summarize (bool): Whether to initialise model and tokenizer.
        """
        self.active = summarize

        if summarize:
            log.info("Initialising CodeT5 model...")
            # Add device detection for GPU acceleration
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(" ----- !! Device is %s" %self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(
                "Salesforce/codet5-base",
                use_fast=True  # Use faster tokenizer implementation
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                "Salesforce/codet5-base-multi-sum"
            ).to(self.device)  # Move model to GPU if available
            log.info(f"Ready! Using device: {self.device}")
        else:
            log.info("Summarization flag not set. Skipping setup.")


    def summarize_function(self, function: Function) -> str:
        """Summarize a function.

        Args:
            function (Function): The function node to summarization.

        Returns:
            str: The summarization
        """
        if not self.model or not self.tokenizer:
            log.warning("No model or tokenizer initialised!")
            return ""

        log.debug(f"Create Docstring node for function `{function.name}`...")
        source_code = clean_source_code(function.source_code)
        return self._summarize_code(source_code)


    def _summarize_code(self, source_code: str) -> str:
        log.debug("Tokenizing...")
        # Add batch processing capability and move to device
        inputs = self.tokenizer(
            source_code,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Add max length to prevent memory issues
        ).to(self.device)

        log.debug("Summarizing...")
        with torch.no_grad():  # Disable gradient calculation for inference
            generated_ids = self.model.generate(
                inputs.input_ids,
                max_length=200,
                num_beams=4,  # Add beam search for better quality
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=2.0
            )

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
