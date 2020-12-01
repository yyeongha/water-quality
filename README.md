
# water-quality

### explanation
```
directory 'gain' is preprocess + gain custom
directory 'tensor2' is gain custom by shevious
```

### let's start project (dev)
```
git clone https://github.com/kotechnia/water-quality
cd water-quality/gain
python -m venv venv

# for linux
. venv/bin/activate

# for windows
venv\Scripts\activate

pip install -r requirements.txt

python debug.py
```

### setting your parameter (dev)
```
parameters_train.json
parameters_test.json
parameters_train_dir.json
```

### excute program in tensor2
```
python main_letter_spam.py --data_name spam --miss_rate 0.2 --batch_size 128 --hint_rate 0.9 --alpha 100 --iterations 10000
```