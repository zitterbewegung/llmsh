from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer
from langchain.chains import ConversationChain
import transformers
import torch
import warnings
warnings.filterwarnings('ignore')
model="codellama/CodeLlama-34b-Instruct-hf"
tokenizer=AutoTokenizer.from_pretrained(model)
pipeline=transformers.pipeline(
     "text-generation",
     model=model,
     tokenizer=tokenizer,
     torch_dtype=torch.bfloat16,
     trust_remote_code=True,
     device_map="auto",
     max_length=1000,
     do_sample=True,
     top_k=10,
     num_return_sequences=1,
     eos_token_id=tokenizer.eos_token_id
     )
llm_local=HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0}) 
