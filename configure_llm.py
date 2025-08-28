# llm_setup.py
import yaml

class LLMWrapper:
    """
    A wrapper class to configure and load different LLM providers 
    based on the settings in config.yaml. 
    Currently supports only 'ctransformers'.
    """
    def __init__(self, llm_config: dict):
        self.config = llm_config
        provider = llm_config.get("provider", "ctransformers") # Default to ctransformers
        
        if provider == "ctransformers":
            self._init_ctransformers()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
    def _init_ctransformers(self):
        """Initialize ctransformers LLM from config.yaml settings."""
        try:
            from langchain_community.llms import CTransformers
        except ImportError as e:
            raise RuntimeError(
                "CTransformers not installed. Install it with `pip install ctransformers`."
            ) from e
            
        cfg = self.config.get("ctransformers", {})
        gen_cfg = cfg.get("config", {})
        
        self.llm = CTransformers(
            model=cfg.get("model_file"), # Path to local GGML/GGUF model file
            model_type=cfg.get("model_type"), # e.g. "llama"
            **gen_cfg # Any additional generation config (max_new_tokens, temperature, etc.)
        )
    
    def invoke(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the response."""
        if hasattr(self.llm, "invoke"):
            return self.llm.invoke(prompt)
        return self.llm(prompt)


# Load config.yaml once and create a global llm instance
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

configured_llm = LLMWrapper(cfg.get("llm"))


## How to use this
# from configure_llm import configured_llm
# llm = configured_llm
# print(llm.invoke('what is your name?'))
