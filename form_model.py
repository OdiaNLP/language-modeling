from wtforms import Form, validators, StringField, IntegerField


class InputFormNgram(Form):
    """Input form class for Odia text generation using ngram LM"""
    prefix = StringField(label='Prefix', default=' ', validators=[validators.InputRequired()])
    max_words = IntegerField(label='Maximum number of words to generate', default=50,
                             validators=[validators.InputRequired()])
