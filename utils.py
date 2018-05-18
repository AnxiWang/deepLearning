from keras.utils import to_categorical
from tqdm import tqdm
from faker import Faker
import random
from babel.dates import format_date
import numpy as np
import keras.backend as K

FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MM YY',
           'd MMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']

fake = Faker()
def load_date():
    dt = fake.date_object()
    try:
        human_readable = format_date(dt, format = random.choice(FORMATS), locale = 'en_US')
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',', '')
        machine_readable = dt.isoformat()
    except AttributeError as e:
        return None, None, None
    return human_readable, machine_readable, dt

def load_dataset(m):
    human_vocab = set()
    machine_vocab = set()
    dataset = []
    Tx = 30

    for i in tqdm(range(m)):
        h, m, _ = load_date()
        if h is not None:
            dataset.append((h, m))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))
    human = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'], list(range(len(human_vocab) + 2))))
    inv_machine = dict(enumerate(sorted(machine_vocab)))
    machine = {v:k for k, v in inv_machine.items()}

    return dataset, human, machine, inv_machine


def string_to_int(string, length, vocab):
    string = string.lower()
    string = string.replace(',', '')

    if len(string) > length:
        string = string[:length]
    rep = list(map(lambda x:vocab.get(x, '<unk>'), string))

    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))

    return rep

def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    X, Y = zip(*dataset) #returns a tuple where the i-th element comes from the i-th iterable argument
    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    Y = [string_to_int(t, Ty, machine_vocab) for t in Y]

    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes = len(human_vocab)), X)))
    Yoh = np.array(list(map(lambda y: to_categorical(y, num_classes = len(machine_vocab)), Y)))

    return X, np.array(Y), Xoh, Yoh

def softmax(x,axis=1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim>2:
        e = K.exp(x - K.max(x,axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e/s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')
