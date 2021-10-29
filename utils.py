import copy
import os.path
from typing import Dict

import dill
import numpy as np
from indicnlp.tokenize.indic_detokenize import trivial_detokenize_indic
from indicnlp.tokenize.indic_tokenize import trivial_tokenize_indic


def generate_ngram_lm(model, inp: str, max_num_words: int) -> Dict[str, str]:
    """Generate text using ngram LM"""

    if max_num_words < 1:
        return {'generation': '', 'message': 'The value of maximum number of words should be positive.'}

    # tokenize
    context = trivial_tokenize_indic(inp)
    if context == ['']:
        num_inp_tokens = 0
        context = None
    else:
        num_inp_tokens = len(context)

    if context is not None and any([word not in model.vocab for word in context]):
        return {'generation': '',
                'message': 'At least one word in the input is out of vocabulary. Try with another input.'}

    if context is None:
        prefix = None
        context = ['<bos>'] * (model.n - 1)
    elif len(context) < model.n - 1:
        prefix = copy.deepcopy(context)
        context = ['<bos>'] * (model.n - 1 - len(context)) + context
    elif len(context) > model.n - 1:
        prefix = copy.deepcopy(context)
        context = context[-(model.n - 1):]
    else:  # len(context) = model.n - 1
        prefix = None

    output = context
    while output[-1] != '<eos>':
        # Form conditional distribution to sample from
        probs, tokens = [], []
        for token in model.count[tuple(context)]:
            p = model.count[tuple(context)][token] / model.total[tuple(context)]
            probs.append(p)
            tokens.append(token)
        # Sample
        wt = np.random.choice(tokens, p=probs)
        output = output + [wt]
        context = context[1:] + [wt]
    if prefix is not None:
        out_tokens = prefix + output[model.n - 1:]
    else:
        out_tokens = output

    # post process

    post_processed_out_tokens = []
    for token in out_tokens:
        if token in ['<bos>', '<eos>']:
            continue
        else:
            post_processed_out_tokens.append(token)

    # trim
    post_processed_out_tokens = post_processed_out_tokens[:num_inp_tokens + max_num_words]

    # detokenize
    detokenized_str = trivial_detokenize_indic(' '.join(post_processed_out_tokens))

    return {'generation': detokenized_str, 'message': ''}


if __name__ == "__main__":
    with open(os.path.join('ngram.lm.pkl'), 'rb') as f:
        _model = dill.loads(f.read())

    print(generate_ngram_lm(model=_model, inp='ପାଖରେ ରାଜ୍ୟ', max_num_words=50, ))
