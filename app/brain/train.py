"""微调意图分类器 — MiniLM 22.7M 版本，CPU ~30分钟"""
import json, os, torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from collections import Counter
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 22.7M，半个GTE的大小
DATA = os.path.join(HERE, "training_data.jsonl")
OUT = os.path.join(HERE, "model_intent")

LABELS = ["casual","emotional_sharing","conflict","ask_fact","recall","request","meta"]
L2ID = {l:i for i,l in enumerate(LABELS)}
ID2L = {i:l for l,i in L2ID.items()}

# data
texts, targets = [], []
with open(DATA, "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line.strip())
        t = d["text"].strip()
        if t and len(t)>=3 and d["intent"] in L2ID:
            texts.append(t); targets.append(L2ID[d["intent"]])
seen = set(); ut,ul = [], []
for t,l in zip(texts,targets):
    if t not in seen: seen.add(t); ut.append(t); ul.append(l)
train_x,val_x,train_y,val_y = train_test_split(ut,ul,test_size=0.15,random_state=42,stratify=ul)
print(f"Train={len(train_x)} Val={len(val_x)}")

# model
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS), id2label=ID2L, label2id=L2ID)

class DS(Dataset):
    def __init__(self, texts, labels, tok):
        self.e = tok(texts, truncation=True, padding=True, max_length=64)
        self.l = labels
    def __len__(self): return len(self.l)
    def __getitem__(self,i): return {"input_ids":torch.tensor(self.e["input_ids"][i]),"attention_mask":torch.tensor(self.e["attention_mask"][i]),"labels":torch.tensor(self.l[i])}

td = DS(train_x,train_y,tok); vd = DS(val_x,val_y,tok)
es = max(1, len(td) // 32)

args = TrainingArguments(output_dir=OUT, eval_strategy="steps", eval_steps=es, save_strategy="steps", save_steps=es, per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=3, learning_rate=3e-5, load_best_model_at_end=True, metric_for_best_model="eval_loss", report_to="none", use_cpu=True, logging_steps=es)

def cm(e): p=e.predictions.argmax(-1); return {"acc":accuracy_score(e.label_ids,p),"f1":f1_score(e.label_ids,p,average="weighted")}

trainer = Trainer(model=model, args=args, train_dataset=td, eval_dataset=vd, compute_metrics=cm)
print(f"\nTraining ({len(train_x)} samples, 3 epochs, batch=16)...")
trainer.train()
r = trainer.evaluate()
print(f"\nFinal: acc={r.get('eval_acc','?')} f1={r.get('eval_f1','?')}")
trainer.save_model(OUT); tok.save_pretrained(OUT)
with open(os.path.join(OUT,"labels.json"),"w") as f: json.dump(LABELS,f,ensure_ascii=False)
print(f"\nDone! model_intent ready.")
