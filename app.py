from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from pytube import extract
# import re


l = []
app = Flask(__name__)

@app.route('/', methods=['POST'])
def hello_world():

    if request.method == "POST":
        # video_id = request.form
        mydict = request.form
        # title = request.form.get('title')
        title = (mydict['video_id'])
        
        
        # extracting id
        title = extract.video_id(title)
        
#         pattern = r"((?<=(v|V)/)|(?<=be/)|(?<=(\?|\&)v=)|(?<=embed/))([\w-]+)"
#         match = re.search(pattern, title)
#         if match:
#             title = match.group()
           


        av = YouTubeTranscriptApi.get_transcript(title , languages=['en'])
        t = []
        for m in av:
            main_txt = (m['text'])
            t.append(main_txt)
            
        # for transcription
        tt = ' '.join(t)
        stt = tt.split(' ')
        vtt=[]
        for s in stt:
            if '[' and 'Music' in s:
                vtt.append("[Music is playing]")
            else:
                vtt.append(s)

        transcription = ' '.join(vtt)

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

        return jsonify({'summary': result, 'transcription': transcription})

    return jsonify({'summary': 'Error ahn mwone'})


if __name__ == "__main__":
    app.run(debug=False)
