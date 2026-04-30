"""
Custom math tool for adding two numbers.
This tool performs arithmetic operations programmatically.
"""

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import logging

logger = logging.getLogger(__name__)


class AddNumbersInput(BaseModel):
    """Input schema for the add_numbers tool."""
    a: float = Field(description="First number to add")
    b: float = Field(description="Second number to add")


class AddNumbersTool(BaseTool):
    """Tool for adding two numbers programmatically."""
    
    name: str = "add_numbers"
    description: str = (
        "Adds two numbers together. "
        "Use this tool when you need to calculate the sum of two numbers. "
        "Input: two numbers (a and b). "
        "Output: the sum of a and b."
    )
    args_schema: Type[BaseModel] = AddNumbersInput
    
    def _run(self, a: float, b: float) -> float:
        """
        Add two numbers programmatically.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            The sum of a and b
        """
        result = a + b
        logger.info(f"Math Tool: {a} + {b} = {result}")
        return result
