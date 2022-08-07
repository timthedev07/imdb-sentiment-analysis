from nltk.corpus import stopwords
import tensorflow as tf
import re
import string

def custom_standardization(input_data):
    stop_words = set(stopwords.words('english'))

    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    stripped_html = tf.strings.regex_replace(stripped_html,r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', ' ')
    stripped_html = tf.strings.regex_replace(stripped_html, r'@([A-Za-z0-9_]+)', ' ' )
    for i in stop_words:
        stripped_html = tf.strings.regex_replace(stripped_html, f' {i} ', " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )
