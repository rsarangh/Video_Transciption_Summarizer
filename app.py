from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


l = []
app = Flask(__name__)

@app.route('/', methods=['POST'])
def hello_world():

    if request.method == "POST":
        # video_id = request.form
        mydict = request.form
        # title = request.form.get('title')
        title = (mydict['video_id'])


        av = YouTubeTranscriptApi.get_transcript(title , languages=['en'])
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
#         print(len(l) == len(sentence_scores))
        N = 5
        top_n_sentences = [l[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]

        result = ' '.join(top_n_sentences)

        return jsonify({'summary': result})

    return jsonify({'summary': 'Error ahn mwone'})


if __name__ == "__main__":
    app.run(debug=True)
