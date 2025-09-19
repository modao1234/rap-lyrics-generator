# Rap Lyrics Generator

This project is a fine-tuned **rap lyrics generator** based on the TinyLlama or Llama 3 models with LoRA adaptation. It allows you to process rap lyrics datasets, fine-tune a base language model, and then generate new rap lyrics conditioned on structured prompts.

---


## Setup

### 1. Create Virtual Environment

### 2. Install Dependencies

### 3. Download Base Model
I use TinyLlama-1.1B-Chat because it is very light, you can use better base model.

---

## Raw Data Downloading
Some website provide free API to download lyrics.

---

## Data Processing

This step converts raw lyrics into JSONL for supervised fine-tuning(I use Gemini API since it has free quota):
```bash
python sft_pack.py --in_dir ./data/raw --out ./data/processed/lyrics.jsonl
```

Each output sample looks like:
```json
{"instruction": "L=8; A=Eminem; P=Verse; TMI=Stage fright, critical moments, self-doubt; Extreme anxiety, panic, tension, shame, frustration; IMG=Physical reactions (sweaty palms/weak knees), vomit, cheap food (mom's pasta), noisy crowds, countdown clocks; rs=AAAA BBBB; Multisyllabic slant rhyme, Internal rhyme, Mosaic/Compound Rhymes, Chain Rhymes", "input": "", "output": "His palms are sweaty, knees weak, arms are heavy\nThere's vomit on his sweater already, mom's spaghetti\nHe's nervous, but on the surface, he looks calm and ready\nTo drop bombs, but he keeps on forgetting\nWhat he wrote down, the whole crowd goes so loud\nHe opens his mouth, but the words won't come out\nHe's chokin', how? Everybody's jokin' now\nThe clock's run out, time's up, over, blaow"}
```

---

## Training

Run fine-tuning with LoRA:
```bash
python train.py
```

Trained checkpoints are saved in `./output/`.

---

## Testing

Generate rap lyrics with the fine-tuned model:
```bash
python test.py
```

Example usage:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./output/llama_rap_lora/checkpoint-100")
model = AutoModelForCausalLM.from_pretrained("./output/llama_rap_lora/checkpoint-100")

prompt = "L=8; A=Eminem; P=Verse; TMI=Homie, Growing up; gratitude, eager, struggle; IMG=Conflict, Ghetto, Gun; rs=ABAB; End rhyme, Multisyllabic rhyme"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---


