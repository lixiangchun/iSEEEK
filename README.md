# iSEEEK
A universal approach for integrating super large-scale single-cell transcriptomes by exploring gene rankings

An example to use the pretrained model.
```python
from transformers import PreTrainedTokenizerFast, BertForMaskedLM
import re

tokenizer = PreTrainedTokenizerFast.from_pretrained("lixiangchun/transcriptome_iseeek_13millioncells_128tokens")
iseeek = BertForMaskedLM.from_pretrained("lixiangchun/transcriptome_iseeek_13millioncells_128tokens")

a = ["B2M MTRNR2L8 UBC FOS TMSB4X UBB FTH1 IFITM1 TPT1 FTL DUSP1", "KRT14 MTRNR2L8 KRT6A B2M GAPDH S100A8 S100A9 KRT5"]

# Replace '-' and '.' with '_'
a = [re.sub(r'\-|\.', '_', s) for s in a]  

batch = tokenizer(a, max_length=128, truncation=True, padding=True, return_tensors="pt")
out = iseeek.bert(**batch)

# [CLS] representation
feature = out.last_hidden_state[:,0,:]

```

