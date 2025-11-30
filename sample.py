import torch
from pathlib import Path
from transformers import AutoTokenizer, GPT2LMHeadModel

from lit_gpt import build_gpt2_config  # 复用你的函数

# 1. 在 outputs/tiny_toy_run 下自动搜 ckpt
base_dir = Path("outputs") / "tiny_toy_run"
ckpt_paths = sorted(base_dir.rglob("*.ckpt"))

if not ckpt_paths:
    raise FileNotFoundError(f"No .ckpt files found under {base_dir}")

# 这里简单用最后一个（通常是最新的），你也可以 print 出来手动选
ckpt_path = ckpt_paths[-1]
print(f"Using checkpoint: {ckpt_path}")

# 2. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. 构建 tiny 模型结构（要和训练时一致）
config = build_gpt2_config(
    model_size="tiny",
    vocab_size=tokenizer.vocab_size,
    n_positions=256,
)
model = GPT2LMHeadModel(config)

# 4. 读取 Lightning ckpt
ckpt = torch.load(ckpt_path, map_location="cpu")
state_dict = ckpt["state_dict"]

# 5. 去掉 "model." 前缀
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("model."):
        new_key = k[len("model.") :]
    else:
        new_key = k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model.eval()

# 6. 生成文本
prompt = "Wikipedia is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

