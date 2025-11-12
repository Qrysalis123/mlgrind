


'''
TOKENIZERS

BPE
word -> chunks -> lookup -> emb -> trainable


~149 813 charactersa
UNICODE

example
    ord("h") -> 104

take unicode data into encodings: utf-8 utf-16 utf-32
utf-8 -> 1 to 4 byts


`str.encode('utf-8')`
'''

list("beep boo poo paaa poo".encode('utf-8'))


"""
but 256 vocab size too small
small context long seq

need large vocab
BPE: compress . merge byte pairs as new token



"""
