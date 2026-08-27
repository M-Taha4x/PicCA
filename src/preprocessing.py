import pandas as pd
from collections import Counter
import re
from torchvision import transforms
#(vocabulary building, tokenization,
# image transforms 
# — reusable helper functions)
def tokenization(caption):
    caption=caption.lower()
    caption=re.sub(r"[a-z]","",caption)
    return caption.split()

def build_vocab(df,min_count=5):
    word_counts=Counter()
    for caption in df['caption']:
        word_counts.update(tokenization(caption))
        
    vocab_words=[word for word ,count in word_counts.items() if count >= min_count]
    special_tokens=['<pad>','<start>','<end>','<unk>']
    vocab=special_tokens + sorted (vocab_words)
    
    word2idx={word:idx for idx,word in enumerate(vocab)}
    idx2word={idx:word for word,idx in word2idx.items()}
    return word2idx,idx2word
def caption_to_sequence(caption,word2idx,max_len=50):
    tokens=tokenization(caption)
    sequence=[word2idx['<start>']]
    sequence+=[word2idx.get(word,word2idx['<unk>']) for word in tokens]
    sequence+=[word2idx['<end>']]
    
    if len(sequence)<max_len:
        sequence+=[word2idx['<pad>']]*(max_len-len(sequence))
    else:
        sequence=sequence[:max_len]
    return sequence

image_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    
])
print("Testing")
