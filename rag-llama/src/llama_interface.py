from llama_cpp import Llama
from typing import List, Optional

class LLaMAInterface:
    """Handles interactions with the LLaMA model."""
    
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: Optional[int] = None):
        """
        Initialize the LLaMA interface.
        
        Args:
            model_path (str): Path to the LLaMA model file
            n_ctx (int): Context window size
            n_threads (int, optional): Number of threads to use for inference
        """
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads
        )
        
    def generate_response(
        self,
        query: str,
        context_chunks: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> str:
        """
        Generate a response based on the query and context chunks.
        
        Args:
            query (str): User's query
            context_chunks (List[str]): Relevant document chunks for context
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter
            
        Returns:
            str: Generated response
        """
        # Construct prompt with context
        context = "\n".join(context_chunks)
        prompt = self._create_prompt(query, context)
        
        # Generate response
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["Human:", "Assistant:"],  # Stop generation at these tokens
            echo=False  # Don't include prompt in output
        )
        
        return response['choices'][0]['text'].strip()
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the LLaMA model.
        
        Args:
            query (str): User's query
            context (str): Context information
            
        Returns:
            str: Formatted prompt
        """
        return f"""Human: Use the following context to answer the question. If you cannot answer this question based on the context alone, say "I cannot answer this question based on the provided context."

Context:
{context}

Question: {query}
