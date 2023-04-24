from flask import Flask, request, jsonify
import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from youtube_transcript_api import YouTubeTranscriptApi

import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


l = []
app = Flask(__name__)




@app.route('/', methods=['POST'])
def hello_world():

    video_id = request.form.get('video_id')

    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    t = []
    for m in transcript:
        main_txt = (m['text'])
        t.append(main_txt)

    linez = []
    for m in t:
        s = list(m)
        ln = []
        for n in s:
            if n == '\xa0' or n == '\n':
                ln.append('')
            else:
                ln.append(n)
        m = ''.join(ln)
        linez.append(m)

    for_this = ' '.join(linez)

    tf_idf_vectorizer = TfidfVectorizer(min_df=2, max_features=None,
                                            strip_accents='unicode',
                                            analyzer='word',
                                            token_pattern=r'\w{1,}',
                                            ngram_range=(1, 3),
                                            use_idf=1, smooth_idf=1,
                                            sublinear_tf=1,
                                            stop_words='english')

    tf_idf_vectorizer.fit(linez)
    sentence_vectors = tf_idf_vectorizer.transform(linez)

    # Getting sentence scores for each sentences
    sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()

    # Sanity checkup
    print(len(linez) == len(sentence_scores))
    N = 5
    top_n_sentences = [linez[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]


    result = ' '.join(top_n_sentences)

    return jsonify({'summary': result})
return jsonify({'summary': 'Error ahn mwone'})


@app.route('/about')
def index():
    return "Hello world about"



if __name__ == "__main__":
    app.run(debug=True)
