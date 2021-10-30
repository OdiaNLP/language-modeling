import os
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from typing import Tuple, Union

import dill
from flask import Flask, render_template, request

from form_model import InputFormNgram
from utils import generate_using_ngram_lm

LM_PATHS = {'ngram': 'ngram.lm.pkl',
            'lstm': 'lstm.lm.pt',
            'transformer': 'transformer.lm.pt'}

# create app
app = Flask(__name__)

# set url postfix
rule = '/generate'


@app.route(rule=rule, methods=['GET', 'POST'])
def index():
    if args.lm == 'ngram':
        form, result = process_ngram_lm()
    else:
        raise NotImplementedError
    return render_template(template_name + '.html', form=form, result=result)


def process_ngram_lm() -> Tuple[InputFormNgram, Union[None, str]]:
    """Process ngram LM"""
    form = InputFormNgram(request.form)
    if request.method == 'POST' and form.validate():
        prefix = form.prefix.data.strip()
        max_words = form.max_words.data

        result = generate_using_ngram_lm(model=model, inp=prefix, max_num_words=max_words)
    else:
        result = None

    if result is not None and responses_path is not None:
        with open(responses_path, 'a', encoding='utf-8') as fr:
            fr.write(
                f'\n\tNEW REQUEST 🤩 @'
                f'{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}\n'
                f'\t[INPUTS] Prefix: {form.prefix.data}, Maximum number of words to generate: {form.max_words.data}\n'
                f'\t[OUTPUTS] Generation: {result["generation"]}, Message: {result["message"]}\n'
            )
    return form, result


if __name__ == '__main__':

    # set template name
    template_name = 'my_view'

    # create responses dir
    os.makedirs('responses', exist_ok=True)

    # specify responses file path
    responses_path = os.path.join('responses', f'{rule[1:]}_logs.txt')

    # command line arguments
    parser = ArgumentParser(description='Run Odia text generation web app')
    parser.add_argument('--lm', type=str, required=True, choices=['ngram', 'lstm', 'transformer'])
    args = parser.parse_args()

    if responses_path is not None:
        with open(responses_path, 'a', encoding='utf-8') as f:
            f.write(
                f'\nstarting app with {args.lm} LM.. '
                f'[{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}]'
                f'\n'
            )

    if args.lm == 'ngram':
        # load ngram language model
        print(f'loading ngram language model from {LM_PATHS[args.lm]}..')
        with open(os.path.join(LM_PATHS[args.lm]), 'rb') as f:
            model = dill.loads(f.read())
        model.total: defaultdict  # this line is necessary to make sure defaultdict does not get removed by reformatting code
    else:
        raise NotImplementedError

    # run app
    app.run(host='127.0.0.1', port=31137, debug=False)
