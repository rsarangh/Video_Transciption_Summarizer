from flask import Flask, render_template, request
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


@app.route('/', methods=["GET", "POST"])
def hello_world():
    # global movie_nameee

    if request.method == "POST":
        myDict = request.form
        # title = request.form.get('title')
        title = (myDict['You-Tube video Id'])

        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
        #model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
        av = YouTubeTranscriptApi.get_transcript(title, languages=['en'])
        t = []
        for m in av:
            main_txt = (m['text'])
            t.append(main_txt)

        l = []
        for m in t:
            s = list(m)
            ln = []
            for n in s:
                if n == '\xa0' or n == '\n':
                    ln.append('')
                else:
                    ln.append(n)
            m = ''.join(ln)
            l.append(m)

        for_this = ' '.join(l)
        #text = for_this
        #tokens = tokenizer(for_this, truncation=True, padding="longest", return_tensors="pt")
        #summary = model.generate(**tokens)


        #
        tf_idf_vectorizer = TfidfVectorizer(min_df=2, max_features=None,
                                            strip_accents='unicode',
                                            analyzer='word',
                                            token_pattern=r'\w{1,}',
                                            ngram_range=(1, 3),
                                            use_idf=1, smooth_idf=1,
                                            sublinear_tf=1,
                                            stop_words='english')

        tf_idf_vectorizer.fit(l)
        sentence_vectors = tf_idf_vectorizer.transform(l)

        # Getting sentence scores for each sentences
        sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()

        # Sanity checkup
        print(len(l) == len(sentence_scores))
        N = 5
        top_n_sentences = [l[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]







        #mm = tokenizer.decode(summary[0])
        #top_n_sentences.append(mm)
        qqq = ' '.join(top_n_sentences)
        # print(qqq)
        return render_template('show.html', inf=qqq)

    return render_template('kd.html')


if __name__ == "__main__":
    app.run(debug=True)