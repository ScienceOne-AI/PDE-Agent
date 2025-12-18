from typing import Any

def create_llm_engine(model_info: str, use_cache: bool = False, is_multimodal: bool = True, **kwargs) -> Any:
    """
    Factory function to create appropriate LLM engine instance.
    """
    if any(x in model_info for x in ["gpt", "o1", "o3", "o4"]):
        from .openai import ChatOpenAI
        return ChatOpenAI(model_info=model_info, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    elif "claude" in model_info:
        from .anthropic import ChatAnthropic
        return ChatAnthropic(model_info=model_info, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    elif any(x in model_info for x in ["deepseek-chat", "deepseek-reasoner"]):
        from .deepseek import ChatDeepseek
        return ChatDeepseek(model_info=model_info, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    # elif "s1" or "S1" in model_info:
    elif any(x in model_info for x in ["S1", "s1"]):
        from .s1 import S1
        return S1(model_info=model_info, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    elif "gemini" in model_info:
        from .gemini import ChatGemini
        return ChatGemini(model_info=model_info, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    elif "grok" in model_info:
        from .xai import ChatGrok
        return ChatGrok(model_info=model_info, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    elif "vllm" in model_info:
        from .vllm import ChatVLLM
        model_info = model_info.replace("vllm-", "")
        return ChatVLLM(model_info=model_info, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    elif "together" in model_info:
        from .together import ChatTogether
        model_info = model_info.replace("together-", "")
        return ChatTogether(model_info=model_info, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    elif "local" in model_info:
        from .local_llm import ChatLocalLLM
        return ChatLocalLLM(model_info=model_info, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
    
    else:
        raise ValueError(f"Engine {model_info} not supported. If you are using Together models, please ensure have the prefix 'together-' in the model string. If you are using VLLM models, please ensure have the prefix 'vllm-' in the model string. For other custom engines, you can edit the factory.py file and add its interface file to add support for your engine. Your pull request will be warmly welcomed!")
    