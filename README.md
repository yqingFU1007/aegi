# AEGI

## Datasets and model weight

Because of the space limitation of anonymous repository, we will release the datasets and model weight upon acceptance.


## Pre-Processing and Extracting the Anchor Events

```bash
cd AEGI 
python extract_anchor_event.py
 ```
## Training the MultiQA retriever

```bash
python main.py --model multiqa --save_to multiqa_multi
```

## Inference

```bash
python main.py --model multiqa --mode eval --load_from multiqa_multi
```

Please explore more argument options in main.py.





