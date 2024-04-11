from os import listdir
from time import time
from nltk import sent_tokenize, word_tokenize, pos_tag, Tree
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.chunk import RegexpParser
from collections import defaultdict
from string import punctuation
from sklearn.feature_extraction import text
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, fbeta_score, precision_recall_curve, precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import math
from tqdm import tqdm
import torch
from transformers import BertModel, BertTokenizer


def tokenization(file):
    s = sent_tokenize(file)
    w = []
    for i in range(len(s)):
        w = w + word_tokenize(s[i])

    # each term needs a tag
    tagged_tokens = pos_tag(w)

    # define a rule for noun phrases: article + adjective + noun*
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    extract = RegexpParser(grammar)
    res = extract.parse(tagged_tokens)

    res_list = []
    for subtree in res:
        if type(subtree) == Tree and subtree.label() == 'NP':
            res_list.append(list(subtree))

    res_final = res_list + [[token] for token in tagged_tokens]

    without_tags = [[w for (w, tag) in lista] for lista in res_final]

    return without_tags


def preprocess(word_tokens, stopwords=[]):
    # Lowercasing, removing stop words and ponctuation
    return_list = []
    ps = PorterStemmer()
    for term in word_tokens:
        filtered_terms = [w for w in term if w.lower() not in stopwords and w.lower() not in punctuation]

        # stemming
        stemmed_terms = [ps.stem(w) for w in filtered_terms]

        if len(stemmed_terms):
            return_list.append(tuple(stemmed_terms))

    return return_list


'''
@input document collection D and optional arguments on text preprocessing
@behavior preprocesses the collection and, using existing libraries, builds an inverted
index with the relevant statistics for the subsequent summarization functions
@output pair with the inverted index I and indexing time
'''


def indexing(D, stopwords_language='english'):
    start_time = time()
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for i, genres in enumerate(tqdm(listdir(D), desc="Indexing Genres")):
        for j, doc in enumerate(listdir(f"{D}/{genres}")):
            if not doc.endswith('.txt'):
                continue

            with open(f"{D}/{genres}/{doc}", 'r') as d:
                # Skip the title
                next(d)
                # tokenization
                tokens = tokenization(d.read())

                # stop words
                stop_words = set(stopwords.words(stopwords_language))

                # removing stop words
                normalized_tokens = preprocess(tokens, stop_words)

                for token in normalized_tokens:
                    index[token][i][j] += 1

    final_index = {word: [(genre_idx, doc_idx, freq) for genre_idx, genre_docs in docs.items() for doc_idx, freq in
                          genre_docs.items()] for word, docs in index.items()}

    return final_index, (time() - start_time)


def folder_id(doc):
    folder_ids = ['business', 'entertainment', 'politics', 'sport', 'tech']
    valores = [0, 1, 2, 3, 4]
    dic = {folder_id: valor for folder_id, valor in zip(folder_ids, valores)}
    for genre in folder_ids:
        if genre in doc:
            return dic[genre]


def file_id(doc):
    path = str(doc)
    parts = path.split('\\')
    return int(parts[-1].split('.')[0]) - 1


def find_dfidf(I, N, d):
    fi_id = file_id(d)
    fo_id = folder_id(d)

    # doc frequency
    df = []
    for term in I:
        df = df + [[term, len(I[term])]]

    filtered = {}
    # select words of selected doc
    for term in list(I.keys()):
        for tupla in I[term]:
            if tupla[0] == fo_id and tupla[1] == fi_id:
                filtered[term] = tupla

    for ele in df:
        if ele[0] not in filtered:
            df.remove(ele)

    idf = {}
    for ele in df:
        idf[ele[0]] = math.log10(N / ele[1])

    tf = {}

    # term frequency of selected docs
    for ele in filtered:
        tf[ele] = filtered[ele][2]

    # 1 + log TF
    for ele in tf:
        tf[ele] = 1 + math.log10(tf[ele])

    tfidf = {key: tf[key] * idf[key] for key in tf}
    return tfidf


def find_BM25(I, N, d, k=1.2, b=0.75):
    fi_id = file_id(d)
    fo_id = folder_id(d)

    # doc frequency - number of documents a term appears
    df = []
    for term in I:
        df = df + [[term, len(I[term])]]

    filtered = {}
    # select words of selected doc
    for term in list(I.keys()):
        for tupla in I[term]:
            if tupla[0] == fo_id and tupla[1] == fi_id:
                filtered[term] = tupla

    for ele in df:
        if ele[0] not in filtered:
            df.remove(ele)

    idf = {}
    for ele in df:
        idf[ele[0]] = math.log10(
            1 + ((N - ele[1] + 0.5) / (ele[1] + 0.5)))  # Same as log(1 + (N - n + 0.5) / (n + 0.5))

    tf = {}
    # term frequency of selected docs
    for ele in filtered:
        tf[ele] = filtered[ele][2]

    avgdl = 0
    for ele in df:
        avgdl = avgdl + ele[1]
    avgdl = avgdl / len(df)

    BM25 = {}
    for ele in tf:
        BM25[ele] = idf[ele] * (tf[ele] * (k + 1)) / (
                tf[ele] + k * (1 - b + b * (len(df) / avgdl)))  # HELP: is len(df) correct?

    return BM25


def get_bert_output(tokenizer, model, sentence, mode='cls', optype='sumsum'):
    tokenized_text = tokenizer.tokenize(sentence)
    tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)])
    segments_tensors = torch.tensor([[1] * len(tokenized_text)])
    outputs = model(tokens_tensor, segments_tensors)
    if mode == 'cls':
        embedding = outputs["last_hidden_state"].squeeze()[0]
    elif mode == 'pooled':
        embedding = outputs["pooler_output"].squeeze()
    else:  # 'hidden'
        layers = torch.stack(outputs['hidden_states'][-4:])
        if optype == "sumsum":
            embedding = torch.sum(layers.sum(0).squeeze(), dim=0)
        elif optype == "summean":
            embedding = torch.sum(layers.mean(0).squeeze(), dim=0)
        elif optype == "meanmean":
            embedding = torch.mean(layers.mean(0).squeeze(), dim=0)
        else:
            embedding = torch.mean(layers.sum(0).squeeze(), dim=0)
    return embedding.detach().numpy()


'''
   @input document d, maximum number of sentences (p) and/or characters (l), order of presentation o (appearance in
    text vs relevance), inverted index I or the collection D, language for stopwords stopwords_language, mechanism to use (dfidf, BM25) 
    scoring, number of documents in the collection (num_docs) 
    @behavior preprocesses d, assesses the relevance of each sentence in d against I according to args, and presents them in
    accordance with p, l and o
    @output summary s of document d, i.e. ordered pairs (sentence position in d, score)

'''


def summarization(d, I, p=8, l=500, o="relevance", stopwords_language='english', scoring="dfidf", num_docs=2225, bert_mode='cls', bert_optype='sumsum', rrf=False, mmr=False, mmr_lambda=0.5):
    
    def reciprocal_rank_fusion(sum_options, y = 5):
        
        optimal = [[x, 0] for x in range(len(sum_options[0]))]
        for opt in sum_options:

            for i, sentence in enumerate(opt):
                optimal[sentence[0]][1] += 1 / (y + i)

        optimal = [tuple(x) for x in optimal]
        return sorted(optimal, key = lambda x: -x[1])
    
    def maximum_marginal_relevance(embedings):

            document = [(i, embedings[i]) for i in range(len(embedings))]
            summary = []

            while len(document) > 0:

                max_mmr = (0, -2, 0)
                for sentence in document:
                    
                    sim_s_d = np.mean([cosine_similarity([sentence[1]], [array])[0, 0] for array in embedings])

                    if summary:
                        sim_s_v = np.mean([cosine_similarity([sentence[1]], [x[2]])[0, 0] for x in summary])
                    else:
                        sim_s_v = 0

                    mmr_score = (1 - mmr_lambda) * sim_s_d - mmr_lambda * sim_s_v
                    if mmr_score > max_mmr[1]:
                        max_mmr = (sentence[0], mmr_score, sentence[1])

                document = [tup for tup in document if tup[0] != max_mmr[0]]
                summary.append(max_mmr)

            return [s[:2] for s in summary]

    # start_time = time()
    with open(d, 'r', encoding="utf-8") as f:
        doc_text = f.read()

    summary = []
    if scoring == "bert" or rrf or mmr:
        sentences = sent_tokenize(doc_text)
        # print(sentences)
        embeddings = []
        for sentence in sentences:
            embeddings.append(get_bert_output(bert_tokenizer, bert_model, sentence, bert_mode, bert_optype))

        if not mmr:
            # Calculate cosine similarity between each sentence and the document
            for i, sentence in enumerate(embeddings):
                score = 0
                for j, other_sentence in enumerate(embeddings):
                    if i != j:
                        score += np.dot(sentence, other_sentence) / (np.linalg.norm(sentence) * np.linalg.norm(other_sentence))
                summary.append((i, score))

        else:
            summary = maximum_marginal_relevance(embeddings)

    # Preprocess sentences
    stopwords_list = set(stopwords.words(stopwords_language))
    preprocessed_sentences = [preprocess(word_tokenize(sentence), stopwords_list) for sentence in sentences]
    if scoring != "dfidf" and scoring != "BM25" and scoring != "bert":
        raise ValueError("Invalid value for 'scoring'. Use 'dfidf', 'bert' or 'BM25'.")

    score_options = []
    if not mmr and (rrf or scoring == "dfidf"):
        score_options.append(find_dfidf(I, num_docs, d))
    if not mmr and (rrf or scoring == "BM25"):
        score_options.append(find_BM25(I, num_docs, d))

    summary_options = []
    if summary:
        summary_options.append(summary)

    for scores in score_options:
        summary = []
        for i, sentence_terms in enumerate(preprocessed_sentences):
            score = 0
            for term in sentence_terms:
                if term in I:
                    if term not in scores:
                        scores[term] = 0
                    score += scores[term]  

            summary.append((i, score))

        if o == "appearance":
            summary = sorted(summary, key=lambda x: x[0])  # Sorts by sentence position
        elif o == "relevance":
            summary = sorted(summary, key=lambda x: x[1], reverse=True)  # Sorts by score, highest first
        else:
            raise ValueError("Invalid value for 'o'. Use 'appearance' or 'relevance'.")
        
        summary_options.append(summary)
    
    optimal_summary = []
    if len(summary_options) > 1:
        optimal_summary = reciprocal_rank_fusion(summary_options)
    else:
        optimal_summary = summary_options[0]

    

    # Find out if the summary should be based on the maximum number of sentences or characters
    selected_sentences = []
    current_length = 0
    for i, (sentence_index, score) in enumerate(optimal_summary):
        sentence = sentences[sentence_index]
        current_length += len(sentence)
        if current_length <= l:
            selected_sentences.append((sentence_index, score))
        else:
            break
        if len(selected_sentences) == p:
            break
    return selected_sentences


'''
    @input document d, maximum number of keywords p, inverted index I and number of documents n
    @behavior extracts the top informative p keywords in d against I according to args
    @output ordered set of p keywords

'''


def keyword_extraction(d, p, I, n=2225):
    tfidf = find_dfidf(I, n, d)

    # normalization
    # vn -> vn^2
    tfidf_square = {}
    for ele in tfidf:
        tfidf_square[ele] = tfidf[ele] ** 2

    # sum of vn^2
    tfidf_sum = 0
    for ele in tfidf_square:
        tfidf_sum = tfidf_sum + tfidf_square[ele]
    # square of sum
    tfidf_square = math.sqrt(tfidf_sum)

    # tfidf / square of sum
    for ele in tfidf:
        tfidf[ele] = tfidf[ele] / tfidf_square

    # ordered normalized values
    norm_ordered = dict(sorted(tfidf.items(), key=lambda item: item[1], reverse=True))

    final_list = []
    for ele in norm_ordered:
        final_list = final_list + [ele]

    return final_list[:p]


def vectorize_references(base_dir):
    references = []

    for genre_id, genre in enumerate(tqdm(listdir(f'{base_dir}/Summaries'), desc="Vectorizing Genres")):
        for doc_id, doc in enumerate(listdir(f"{base_dir}/Summaries/{genre}")):

            total_sentences = []
            ref_sentences = []
            with open(f"{base_dir}/News Articles/{genre}/{doc}", 'r') as f:
                total_sentences = sent_tokenize(f.read())

            with open(f"{base_dir}/Summaries/{genre}/{doc}", 'r') as f:
                ref_sentences = sent_tokenize(f.read())

            common = ()
            for i, sentence in enumerate(total_sentences):
                for ref in ref_sentences:
                    if sentence in ref:
                        common += (i,)
                        break

            references.append(((genre_id, doc_id), common))

    return references


'''
    @input the set of summaries S set, the corresponding reference extracts Rset
    @behavior assesses the produced summaries against the reference ones using the tar-get evaluation criteria
    @output evaluation statistics, including F-measuring at predefined p-or-l summary limits, recall-and-precision curves, MAP, and efficiency

'''


def evaluation(S, Rset, beta=1.75):
    def plot_pr_curve(predictions, answers, title=""):

        # precision, recall, thresholds = precision_recall_curve(answers, predictions)

        plt.figure()

        # Compute precision-recall pairs for each document
        for i in range(len(answers)):
            precision, recall, t = precision_recall_curve(answers[i], predictions[i])
            plt.plot(recall, precision, marker='x', markersize=10)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if not title:
            plt.title('Precision-Recall Curve')
        else:
            plt.title(f"Precision-Recall Curve for Genre ID = {title}")
        plt.show()

    def collapse_scores(summary_clp):
        return tuple([x[0] for x in summary_clp])

    def binarize_sets(relevant_summaries, relevant_references):

        max_sentences = 0
        for i in relevant_summaries + relevant_references:
            if max(i) > max_sentences:
                max_sentences = max(i)

        def tuple_to_binary_labels(tuple_list, max_sentences):
            binary_labels = np.zeros(max_sentences, dtype=int)
            binary_labels[np.array(tuple_list) - 1] = 1
            return binary_labels

        binary_summaries = np.array([tuple_to_binary_labels(t, max_sentences) for t in relevant_summaries])
        binary_references = np.array([tuple_to_binary_labels(t, max_sentences) for t in relevant_references])

        return binary_summaries, binary_references

    # Filtering to make sure references exist in the summaries
    sum_indexes = [summ[0] for summ in S]
    Rset = [ref for ref in Rset if ref[0] in sum_indexes]

    total_fscore = 0
    total_precision = 0
    total_auc = 0
    for genre_id in range(5):
        relevant_summaries = [collapse_scores(s[1]) for s in S if s[0][0] == genre_id]
        relevant_references = [s[1] for s in Rset if s[0][0] == genre_id]
        binary_summaries, binary_references = binarize_sets(relevant_summaries, relevant_references)

        f_score = fbeta_score(binary_references, binary_summaries, beta=beta, average="micro")
        precision = precision_score(binary_references, binary_summaries, average="micro")
        auc_score = roc_auc_score(binary_references, binary_summaries, average="micro")
        total_fscore += f_score
        total_precision += precision
        total_auc += auc_score

        print(f"Genre ID = {genre_id}")
        print(f"\tF-Measure with β = {beta} --> {f_score}")
        print(f"\tPrecision --> {precision}")
        print(f"\tAUC --> {auc_score}")

        plot_pr_curve(binary_summaries, binary_references, title=str(genre_id))

    # Pre-processing the summaries and references
    S = [collapse_scores(s[1]) for s in S]
    Rset = [s[1] for s in Rset]
    binary_S, binary_Rset = binarize_sets(S, Rset)

    print(f"Average F-Score for all the categories --> {round(total_fscore / 5, 4)}")
    print(f"The Mean Average Precision (MAP) -->  {round(total_precision / 5, 4)}")
    print(f"The Average Area Under the ROC Curve (AUC) -->  {round(total_auc / 5, 4)}")
    plot_pr_curve(binary_S, binary_Rset, "All")


inp_collection = r"BBC News Summary\News Articles"
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True, do_lower_case=True)
bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
