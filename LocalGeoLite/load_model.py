from typing import Optional, Tuple, Dict
from unsloth import FastLanguageModel
from transformers import PreTrainedModel, PreTrainedTokenizer
import os
import torch
import gc
from IPython import get_ipython

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ModelManager:

    _instance = None
    _model: Optional[PreTrainedModel] = None
    _tokenizer: Optional[PreTrainedTokenizer] = None
    _code = None
    _text = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def load_model(
            cls,
            cache_dir: Optional[str] = None,
            max_seq_length: int = 2048,
            dtype=None,
            load_in_4bit: bool = True
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        # 如果已有加载的模型实例，直接返回
        if cls._model is not None and cls._tokenizer is not None:
            print("✅ Found that the model has aleady been initialized.")
            return cls._model, cls._tokenizer

        # 设置缓存目录
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir

        print("🎯 Loading new model instance\n🟠 Downloading takes some time...")
        # 加载新的模型实例
        cls._model, cls._tokenizer = FastLanguageModel.from_pretrained(
            model_name="JackyMu/LocalGeoLite",
            cache_dir=cache_dir,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # 启用快速推理
        FastLanguageModel.for_inference(cls._model)

        from .code_generator import code
        from .text_generator import text

        cls._code = code(cls._model, cls._tokenizer)
        cls._text = text(cls._model, cls._tokenizer)

    @classmethod
    def unload_model(cls) -> None:
        try:
            if get_ipython() is not None:
                print("🔄 Running in Jupyter. For complete GPU memory clearance, please restart the kernel.")
                print("ℹ️ Tip: You can restart the kernel by clicking 'Kernel' -> 'Restart' in the menu.")
                return
        except:
            pass

        if cls._model is not None:
            if hasattr(cls._model, 'cpu'):
                cls._model.cpu()
            # 删除模型
            del cls._model
            cls._model = None
            
        if cls._tokenizer is not None:
            del cls._tokenizer
            cls._tokenizer = None
            
        if cls._code is not None:
            del cls._code
            cls._code = None
            
        if cls._text is not None:
            del cls._text
            cls._text = None

        # 强制进行垃圾回收
        gc.collect()
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            
        print("🗑️ Model unloaded and GPU memory cleared.")


# 创建ModelManager实例
_model_manager = ModelManager()


def load_model(
        cache_dir: Optional[str] = None,
        max_seq_length: int = 2048,
        dtype=None,
        load_in_4bit: bool = True
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    return _model_manager.load_model(
        cache_dir=cache_dir,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )


def unload_model() -> None:
    _model_manager.unload_model()


# 全局code和text函数
def code(prompt: str, max_new_tokens: int = 256) -> str:
    if _model_manager._code is None:
        raise RuntimeError("Model not loaded. Please call load_model() first.")
    return _model_manager._code(prompt, max_new_tokens)


def text(prompt: str, max_new_tokens: int = 256) -> str:
    if _model_manager._text is None:
        raise RuntimeError("Model not loaded. Please call load_model() first.")
    return _model_manager._text(prompt, max_new_tokens)
