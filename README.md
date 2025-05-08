## Requirements


We test the codes in the following environments:

- Python==3.6.9
- PyTorch==1.3.0
- Torchvision==0.4.1


## Dataset


Change the path to CelebA like the following:

```
├── datas
│   ├── img_align_celeba      
│   ├── train_40_att_list.txt       
│   └── test_40_att_list.txt

```

## Training

To train on CelebA:

```bash
python main.py
```
