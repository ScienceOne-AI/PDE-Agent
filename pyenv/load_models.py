from pathlib import Path
import os
from transformers import AutoTokenizer, AutoModel

model_name = "tbs17/MathBERT"
save_dir = "./tmp"

my_dir = Path(save_dir)
my_dir.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(my_dir)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=my_dir)
model     = AutoModel.from_pretrained(model_name,     cache_dir=my_dir)

print("embedding models are saved in：", my_dir.resolve())