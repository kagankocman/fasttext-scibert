import fasttext.util
import nltk
import numpy as np
import tokenizer
import torch
import os
import re
from transformers import AutoTokenizer, AutoModel
from flask import request, redirect, url_for, Flask, render_template
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

uri = ("**********")

client = MongoClient(uri, server_api=ServerApi('1'))
# MongoDB'ye bağlan
users_collection = client.mydatabase.users
scibert_vectors_collection = client.mydatabase.scibert_vectors

article_data = {}
data_folder = "Krapivin2009\\docsutf8"
keys_folder = "Krapivin2009\\keys"

app = Flask(__name__)

def read_krapivin():
    for filename in os.listdir(data_folder):
        # Dosya yolu oluşturun
        filepath = os.path.join(data_folder, filename)
        # Dosya ismini makale ID'si olarak alın
        article_id = os.path.splitext(filename)[0]

        # Dosyayı açarak başlık ve özeti alın
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            # Başlık kısmını --T ve --A arasındaki metin olarak alın
            title_start = content.find('--T') + 3  # --T'den sonraki karakterler
            title_end = content.find('--A')  # --A'nın öncesindeki karakterler
            title = content[title_start:title_end].strip()

            # Özet kısmını --A ve --B arasındaki metin olarak alın
            abstract_start = content.find('--A') + 3  # --A'dan sonraki karakterler
            abstract_end = content.find('--B')  # --B'nin öncesindeki karakterler
            abstract = content[abstract_start:abstract_end].strip()

        # Başlık ve özet verilerini sözlüğe ekleyin
        article_data[article_id] = {'title': title, 'abstract': abstract}

def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

read_krapivin()
setup_nltk()

# load fasttext
fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')

#load scibert
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

#fasttext vector
user_vectors_fasttext = []
#scibert vector
user_vectors_scibert = []

article_vectors_scibert = []
for vector_document in scibert_vectors_collection.find():
    article_id = vector_document['id']
    article_vector = np.array(vector_document['vector'])
    article_vectors_scibert.append({
        'id': article_id,
        'vector': article_vector
    })

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['mail']
        password = request.form['psw']

        user = users_collection.find_one({'email': email, 'password': password})

        if user:
            # Kullanıcı varsa, mainpage'e yönlendir ve email'i URL'de gönder
            return redirect(url_for('mainpage', email=email))
        else:
            return render_template('login.html')

    elif request.method == 'GET':
        return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Formdan gelen verileri al
        name = request.form['name']
        surname = request.form['surname']
        email = request.form['email']
        password = request.form['password']
        dob = request.form['dob']
        interests = request.form['interests']

        # Yeni kullanıcıyı oluştur
        new_user = {
            'name': name,
            'surname': surname,
            'email': email,
            'password': password,
            'dob': dob,
            'interests': interests
        }

        # Kullanıcıyı MongoDB'ye ekle
        users_collection.insert_one(new_user)

        # Kayıt işlemi tamamlandıktan sonra ana sayfaya yönlendir
        return redirect(url_for('login'))
    elif request.method == 'GET':
        return render_template('register.html')

@app.route('/mainpage/<email>')
def mainpage(email):
    # Kullanıcı adı ve soyadını MongoDB'den al
    user = users_collection.find_one({'email': email})
    name = user.get('name', '')
    surname = user.get('surname', '')

    # mainpage'e kullanıcının email'ini, adını ve soyadını gönder
    return render_template('mainpage.html', email=email, name=name, surname=surname)

@app.route('/mainpage/<email>/profile', methods=['GET', 'POST'])
def profile(email):
    if request.method == 'POST':
        # Formdan gelen verileri al
        name = request.form['name']
        surname = request.form['surname']
        password = request.form['password']
        dob = request.form['dob']
        interests = request.form['interests']

        # Kullanıcıyı güncelle
        query = {'email': email}
        new_values = {'$set': {'name': name, 'surname': surname, 'password': password, 'dob': dob, 'interests': interests}}
        users_collection.update_one(query, new_values)

        # Profil güncellendikten sonra mainpage'e yönlendir
        return redirect(url_for('mainpage', email=email))
    else:
        # Kullanıcı adı, soyadı, doğum tarihi ve ilgi alanlarını MongoDB'den al
        user = users_collection.find_one({'email': email})
        name = user.get('name', '')
        surname = user.get('surname', '')
        dob = user.get('dob', '')
        interests = user.get('interests', '')

        # Profil sayfasına kullanıcının bilgilerini gönder
        return render_template('profile.html', email=email, name=name, surname=surname, dob=dob, interests=interests)

@app.route('/mainpage/<email>/search', methods=['GET', 'POST'])
def search(email):
    if request.method == 'POST':
        return redirect(url_for('search_results', email=email, query=request.form['query']))
    else:
        return render_template('search.html', email=email)

@app.route('/mainpage/<email>/search/<query>')
def search_results(email, query):
    # Aranan kelimenin içerisinde olduğu makalelerin listesi
    search_results = []

    # Her bir makaleyi kontrol et
    for article_id, data in article_data.items():
        # Makale başlığı ve özetinde aranan kelimeyi ara (büyük/küçük harf duyarlı olmayacak şekilde)
        if query.lower() in data['title'].lower() or query.lower() in data['abstract'].lower():
            # Eğer aranan kelime makale başlığında veya özetinde geçiyorsa, bu makaleyi arama sonuçları listesine ekle
            search_results.append({'article_id': article_id, 'title': data['title'], 'abstract': data['abstract']})

    # Arama sonuçlarını HTML şablonuna gönder
    return render_template('search_results.html', email=email, query=query, search_results=search_results)

@app.route('/mainpage/<email>/suggestion', methods=['GET', 'POST'])
def suggestion(email):
    if request.method == 'POST':
        selected_articles_fasttext = request.form.getlist('selected_articles_fasttext')
        selected_articles_scibert = request.form.getlist('selected_articles_scibert')
        #selectedları db' ekle fasttext
        selected_articles_with_similarity_fasttext = []
        for selected_article in selected_articles_fasttext:
            for recommended_article in users_collection.find_one({'email': email}).get('recommended_articles_fasttext',
                                                                                       []):
                if recommended_article['id'] == selected_article:
                    similarity_float = float(recommended_article['similarity'])
                    selected_articles_with_similarity_fasttext.append({
                        'id': recommended_article['id'],
                        'name': recommended_article['name'],
                        'similarity': similarity_float
                    })
                    break

        try:
            users_collection.update_one(
                {'email': email},
                {'$push': {'selected_articles_fasttext': {'$each': selected_articles_with_similarity_fasttext}}}
            )
        except Exception as e:
            print("Error updating selected articles:", e)

        #selectedları db'e ekle scibert
        selected_articles_with_similarity_scibert = []
        for selected_article in selected_articles_scibert:
            for recommended_article in users_collection.find_one({'email': email}).get('recommended_articles_scibert',
                                                                                       []):
                if recommended_article['id'] == selected_article:
                    similarity_float = float(recommended_article['similarity'])
                    selected_articles_with_similarity_scibert.append({
                        'id': recommended_article['id'],
                        'name': recommended_article['name'],
                        'similarity': similarity_float
                    })
                    break

        try:
            users_collection.update_one(
                {'email': email},
                {'$push': {'selected_articles_scibert': {'$each': selected_articles_with_similarity_scibert}}}
            )
        except Exception as e:
            print("Error updating selected articles:", e)

        # fasttext En benzer 5 makale için özet bilgilerin toplanması
        selected_fasttextarticles_abstracts = []
        for article in selected_articles_fasttext:
            article_info = article_data[article]
            selected_fasttextarticles_abstracts.append(article_info.get('abstract'))

        # fasttext uservector güncellemesi
        selected_fasttextarticle_vectors = []
        for abstract in selected_fasttextarticles_abstracts:
            if abstract:
                # Abstract'in işlenmesi
                abstract = abstract.lower()
                abstract = re.sub(r'[^\w\s]', '', abstract)
                words = word_tokenize(abstract)
                filtered_words = [word for word in words if word not in stop_words]
                stemmed_words = [stemmer.stem(word) for word in filtered_words]
                processed_text = ' '.join(stemmed_words)

                # fastText modeli ile vektör temsilinin oluşturulması
                article_vector = ft.get_sentence_vector(processed_text)
                selected_fasttextarticle_vectors.append(article_vector)

        # scibert uservector güncellemesi
        selected_scibertarticle_vectors = []
        # selected_articles_scibert listesindeki her makale için
        for selected_article in selected_articles_scibert:
            # article_vectors_scibert listesindeki her makale için
            for scibert_article in article_vectors_scibert:
                # Eğer makale id'si selected_article ile eşleşiyorsa
                if scibert_article['id'] == selected_article:
                    # article_vector'u al
                    article_vector = scibert_article['vector']
                    # selected_scibertarticle_vectors listesine ekle
                    selected_scibertarticle_vectors.append(article_vector)
                    # Eşleşmeyi bulduktan sonra iç döngüden çık
                    break

        # Seçilen makalelerin vektörlerinin ortalamasını alın
        selected_fasttextarticles_average_vector = sum(selected_fasttextarticle_vectors) / len(selected_fasttextarticle_vectors)

        # Seçilen makalelerin vektörlerinin ortalamasını alın
        selected_scibertarticle_average_vector = np.mean(selected_scibertarticle_vectors, axis=0)

        # Kullanıcının vektörü ve seçilen makalelerin ortalama vektörünü birleştirerek yeni kullanıcı vektörünü oluşturun
        user_avgvector_fasttext = sum(user_vectors_fasttext) / len(user_vectors_fasttext)
        user_avgvector_fasttext = (user_avgvector_fasttext + selected_fasttextarticles_average_vector) / 2

        # Kullanıcının vektörü ve seçilen makalelerin ortalama vektörünü birleştirerek yeni kullanıcı vektörünü oluşturun
        user_avgvector_scibert = sum(user_vectors_scibert) / len(user_vectors_scibert)
        user_avgvector_scibert = (user_avgvector_scibert + selected_scibertarticle_average_vector) / 2

        # similarity fasttext
        article_similarities_fasttext = []
        for article_id, data in article_data.items():
            title = data.get('title')
            abstract = data.get('abstract')
            if title and abstract:
                # Abstract'in işlenmesi
                abstract = abstract.lower()
                abstract = re.sub(r'[^\w\s]', '', abstract)
                words = word_tokenize(abstract)
                filtered_words = [word for word in words if word not in stop_words]
                stemmed_words = [stemmer.stem(word) for word in filtered_words]
                processed_text = ' '.join(stemmed_words)

                # fastText modeli ile vektör temsilinin oluşturulması
                article_vector = ft.get_sentence_vector(processed_text)

                # Cosine similarity hesaplanması
                similarity = cosine_similarity([user_avgvector_fasttext], [article_vector])[0][0]
                article_similarities_fasttext.append((article_id, float(similarity)))

        # Benzerliklere göre makale başlıklarının sıralanması
        sorted_articles_fasttext = sorted(article_similarities_fasttext, key=lambda x: x[1], reverse=True)

        # Kullanıcıya daha önce önerilen makaleleri alın
        user = users_collection.find_one({'email': email})
        previously_recommended_fasttextarticles = user.get('recommended_articles_fasttext', [])
        previously_recommended_fasttextarticle_ids = {article['id'] for article in previously_recommended_fasttextarticles}

        # Daha önce önerilen makaleleri çıkar
        sorted_articles_fasttext = [article for article in sorted_articles_fasttext if
                                    article[0] not in previously_recommended_fasttextarticle_ids]

        # Kullanıcının belgesine yeni alan ekleme
        recommended_articles_fasttext = []
        for article_id, similarity in sorted_articles_fasttext[:5]:
            article_info = article_data[article_id]
            recommended_articles_fasttext.append({
                'id': article_id,
                'name': article_info.get('title'),
                'similarity': similarity
            })

        # recommended_articles_fasttext alanına yazma
        users_collection.update_one(
            {'email': email},
            {'$push': {'recommended_articles_fasttext': {'$each': recommended_articles_fasttext}}}
        )

        # En benzer 5 makale için gerekli bilgilerin toplanması
        top_articles_fasttext = []
        for article_id, similarity in sorted_articles_fasttext[:5]:
            article_info = article_data[article_id]
            top_articles_fasttext.append({
                'id': article_id,
                'title': article_info.get('title'),
                'abstract': article_info.get('abstract'),
                'similarity': similarity
            })

        article_similarities_scibert = []
        # article_data içindeki her makale için benzerlik hesaplaması
        for article_id, data in article_data.items():
            title = data.get('title')
            abstract = data.get('abstract')
            if title and abstract:
                # Makale vektörünü article_vectors listesinden çekme
                article_vector = None
                for article in article_vectors_scibert:
                    if article['id'] == article_id:
                        article_vector = article['vector']
                        break

                if article_vector is not None:
                    # Cosine similarity hesaplanması
                    similarity = cosine_similarity(user_avgvector_scibert.reshape(1, -1), article_vector.reshape(1, -1))[0][0]
                    article_similarities_scibert.append((article_id, float(similarity)))

        # Benzerliklere göre makale isimlerinin sıralanması
        sorted_articles_scibert = sorted(article_similarities_scibert, key=lambda x: x[1], reverse=True)

        # Kullanıcıya daha önce önerilen makaleleri alın
        user = users_collection.find_one({'email': email})
        previously_recommended_scibertarticles = user.get('recommended_articles_scibert', [])
        previously_recommended_scibertarticle_ids = {article['id'] for article in
                                                      previously_recommended_scibertarticles}

        # Daha önce önerilen makaleleri çıkar
        sorted_articles_scibert = [article for article in sorted_articles_scibert if
                                    article[0] not in previously_recommended_scibertarticle_ids]

        # Kullanıcının belgesine yeni alan ekleme
        recommended_articles_scibert = []
        for article_id, similarity in sorted_articles_scibert[:5]:
            article_info = article_data[article_id]
            recommended_articles_scibert.append({
                'id': article_id,
                'name': article_info.get('title'),
                'similarity': similarity
            })

        # recommended_articles_scibert alanına yazma
        users_collection.update_one(
            {'email': email},
            {'$push': {'recommended_articles_scibert': {'$each': recommended_articles_scibert}}}
        )

        # En benzer 5 makale için gerekli bilgilerin toplanması
        top_articles_scibert = []
        for article_id, similarity in sorted_articles_scibert[:5]:
            article_info = article_data[article_id]
            top_articles_scibert.append({
                'id': article_id,
                'title': article_info.get('title'),
                'abstract': article_info.get('abstract'),
                'similarity': similarity
            })

        #precision değeri hesaplama fasttext
        user_data = users_collection.find_one({"email": email})
        TP_ft = len(user_data.get("selected_articles_fasttext", []))
        FP_ft = len(user_data.get("recommended_articles_fasttext", [])) - 5

        if FP_ft == 0:
            precision_fasttext = 0
        else:
            precision_fasttext = TP_ft / FP_ft * 100

        # precision değeri hesaplama scibert
        user_dt = users_collection.find_one({"email": email})
        TP_sb = len(user_dt.get("selected_articles_scibert", []))
        FP_sb = len(user_dt.get("recommended_articles_scibert", [])) - 5

        if FP_sb == 0:
            precision_scibert = 0
        else:
            precision_scibert = TP_sb / FP_sb * 100

        return render_template('suggestion.html', email=email, top_articles_fasttext=top_articles_fasttext,
                               top_articles_scibert=top_articles_scibert, precision_fasttext=precision_fasttext, precision_scibert=precision_scibert)

    else:
        # db'den interestleri çekip user_keywords dizisine ekliyoruz
        user = users_collection.find_one({'email': email})
        user_input = user.get('interests', '')
        user_keywords = user_input.split('\n')
        user_keywords = list(filter(None, [item.strip() for item in user_keywords]))


        #fasttext
        for user_keyword in user_keywords:
            # FastText modeli ile ilgi alanının vektörünün oluşturulması
            user_keyword_vector = ft.get_sentence_vector(user_keyword)
            user_vectors_fasttext.append(user_keyword_vector)

        # Tüm anahtar kelime vektörlerinin bir araya getirilmesi
        user_avgvector_fasttext = sum(user_vectors_fasttext) / len(user_vectors_fasttext)

        # benzer fasttext
        article_similarities_fasttext = []

        for article_id, data in article_data.items():
            title = data.get('title')
            abstract = data.get('abstract')
            if title and abstract:
                # Abstract'in işlenmesi
                abstract = abstract.lower()
                abstract = re.sub(r'[^\w\s]', '', abstract)
                words = word_tokenize(abstract)
                filtered_words = [word for word in words if word not in stop_words]
                stemmed_words = [stemmer.stem(word) for word in filtered_words]
                processed_text = ' '.join(stemmed_words)

                # fastText modeli ile vektör temsilinin oluşturulması
                article_vector = ft.get_sentence_vector(processed_text)

                # Cosine similarity hesaplanması
                similarity = cosine_similarity([user_avgvector_fasttext], [article_vector])[0][0]
                article_similarities_fasttext.append((article_id, float(similarity)))

        # Benzerliklere göre makale başlıklarının sıralanması
        sorted_articles_fasttext = sorted(article_similarities_fasttext, key=lambda x: x[1], reverse=True)

        # Kullanıcının belgesine yeni alan ekleme
        recommended_articles_fasttext = []
        for article_id, similarity in sorted_articles_fasttext[:5]:
            article_info = article_data[article_id]
            recommended_articles_fasttext.append({
                'id': article_id,
                'name': article_info.get('title'),
                'similarity': similarity
            })

        # recommended_articles_fasttext alanına yazma
        users_collection.update_one(
            {'email': email},
            {'$set': {'recommended_articles_fasttext': recommended_articles_fasttext}}
        )

        # En benzer 5 makale için gerekli bilgilerin toplanması
        top_articles_fasttext = []
        for article_id, similarity in sorted_articles_fasttext[:5]:
            article_info = article_data[article_id]
            top_articles_fasttext.append({
                'id': article_id,
                'title': article_info.get('title'),
                'abstract': article_info.get('abstract'),
                'similarity': similarity
            })

        #scibert
        for user_keyword in user_keywords:
            # Anahtar kelimenin tokenlerine ayırılması ve SciBERT ile vektörlerinin alınması
            encoded_input = tokenizer(user_keyword.strip(), return_tensors='pt')
            with torch.no_grad():
                user_keyword_vector = model(**encoded_input).last_hidden_state.mean(dim=1)
            user_vectors_scibert.append(user_keyword_vector.numpy())

        # Tüm anahtar kelime vektörlerinin bir araya getirilmesi ve kullanıcı vektörünün oluşturulması
        user_vector = np.mean(user_vectors_scibert, axis=0)

        article_similarities_scibert = []
        # article_data içindeki her makale için benzerlik hesaplaması
        for article_id, data in article_data.items():
            title = data.get('title')
            abstract = data.get('abstract')
            if title and abstract:
                # Makale vektörünü article_vectors listesinden çekme
                article_vector = None
                for article in article_vectors_scibert:
                    if article['id'] == article_id:
                        article_vector = article['vector']
                        break

                if article_vector is not None:
                    # Cosine similarity hesaplanması
                    similarity = cosine_similarity(user_vector.reshape(1, -1), article_vector.reshape(1, -1))[0][0]
                    article_similarities_scibert.append((article_id, float(similarity)))

        # Benzerliklere göre makale isimlerinin sıralanması
        sorted_articles_scibert = sorted(article_similarities_scibert, key=lambda x: x[1], reverse=True)

        # Kullanıcının belgesine yeni alan ekleme
        recommended_articles_scibert = []
        for article_id, similarity in sorted_articles_scibert[:5]:
            article_info = article_data[article_id]
            recommended_articles_scibert.append({
                'id': article_id,
                'name': article_info.get('title'),
                'similarity': similarity
            })

        # recommended_articles_scibert alanına yazma
        users_collection.update_one(
            {'email': email},
            {'$set': {'recommended_articles_scibert': recommended_articles_scibert}}
        )

        # En benzer 5 makale için gerekli bilgilerin toplanması
        top_articles_scibert = []
        for article_id, similarity in sorted_articles_scibert[:5]:
            article_info = article_data[article_id]
            top_articles_scibert.append({
                'id': article_id,
                'title': article_info.get('title'),
                'abstract': article_info.get('abstract'),
                'similarity': similarity
            })

        return render_template('suggestion.html', email=email, top_articles_fasttext=top_articles_fasttext,
                               top_articles_scibert=top_articles_scibert, precision_fasttext=0, precision_scibert=0)

if __name__ == '__main__':
    app.run()