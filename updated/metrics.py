from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import defaultdict
import nltk
import math 
nltk.download('punkt')

# ROUGE-L 指标计算函数
def rouge_l(reference, candidate):
    reference = nltk.word_tokenize(reference.lower())
    candidate = nltk.word_tokenize(candidate.lower())
    lcs_matrix = [[0] * (len(candidate) + 1) for _ in range(len(reference) + 1)]
    for i in range(1, len(reference) + 1):
        for j in range(1, len(candidate) + 1):
            if reference[i - 1] == candidate[j - 1]:
                lcs_matrix[i][j] = lcs_matrix[i - 1][j - 1] + 1
            else:
                lcs_matrix[i][j] = max(lcs_matrix[i - 1][j], lcs_matrix[i][j - 1])
    lcs = lcs_matrix[-1][-1]
    precision = lcs / len(candidate) if candidate else 0
    recall = lcs / len(reference) if reference else 0
    if precision + recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)

# 结合后的 CIDEr-D 指标计算函数
def cider_d(references, candidates):
    def tokenize(text):
        # 简单的分词函数，可根据实际需求替换为更复杂的分词器
        return text.lower().split()

    def count_ngrams(tokens, n):
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i + n])
            ngrams[ngram] += 1
        return ngrams

    # 计算文档频率（DF）
    df = defaultdict(int)
    for ref in references:
        ref_tokens = tokenize(ref)
        for n in range(1, 5):
            ref_ngrams = count_ngrams(ref_tokens, n)
            for ngram in ref_ngrams:
                df[ngram] += 1

    # 计算候选文本与参考文本的相似度得分
    scores = []
    for cand in candidates:
        cand_tokens = tokenize(cand)
        cand_score = 0
        for n in range(1, 5):
            cand_ngrams = count_ngrams(cand_tokens, n)
            common_ngrams = set(cand_ngrams.keys()) & set(df.keys())
            tf_idf_sum = 0
            for ngram in common_ngrams:
                # 计算 TF-IDF 值
                tf = cand_ngrams[ngram]
                idf = math.log(len(references) / df[ngram])
                tf_idf_sum += tf * idf
            # 归一化得分
            if len(common_ngrams) > 0:
                cand_score += tf_idf_sum / len(common_ngrams)
        scores.append(cand_score)

    # 计算平均得分及标准差
    mean_score = sum(scores) / len(scores)
    return mean_score