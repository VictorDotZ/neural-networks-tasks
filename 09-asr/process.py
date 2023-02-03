import pickle

import pandas as pd

import torch
from mpire import WorkerPool

from ctc_decoder import CTCDecoder
from language_model import LanguageModel

from metrics import (
    normalize_text,
    relative_cer,
    oracle_relative_cer,
    relative_wer,
    oracle_relative_wer,
)


TEST_CTC_DATASET_PATH = "./test_data.pt"
VOCAB_PKL_PATH = "./vocab.pkl"

dataset = torch.load(TEST_CTC_DATASET_PATH)
with open(VOCAB_PKL_PATH, "rb") as fin:
    vocab = pickle.load(fin)

corpus = []

for _, ground_truth in dataset.values():
    corpus.append(ground_truth)

corpus = " ".join(corpus)

alphabet = []
for char_id in vocab:
    if char_id > 4:
        alphabet.append(vocab[char_id])

alphabet = "".join(alphabet)

lm = LanguageModel(corpus, alphabet)

ctc_decoder = CTCDecoder(vocab, lm)


def beam_search(logits, beam_size_):
    return ctc_decoder.beam_search_decode(logits[0], beam_size_)


if __name__ == "__main__":
    graph_results = {"oracle_wer": [], "oracle_cer": [], "top1_wer": [], "top1_cer": []}
    beam_sizes = [4]
    for beam_size in beam_sizes:
        top1_wer, top1_cer, oracle_wer, oracle_cer = 0, 0, 0, 0

        with WorkerPool(n_jobs=8) as pool:
            results = pool.map(
                beam_search,
                [(x[0], beam_size) for x in dataset.values()],
                progress_bar=True,
            )

        for (_, ground_truth), predicted in zip(dataset.values(), results):

            ground_truth = normalize_text(ground_truth)
            # из бимсерча получается список гипотез
            predicted = [normalize_text(pred) for pred in predicted]

            oracle_cer += oracle_relative_cer(ground_truth, predicted)
            oracle_wer += oracle_relative_wer(ground_truth, predicted)

            # [0] -- наиболее вероятный луч
            top1_cer += relative_cer(ground_truth, predicted[0])
            top1_wer += relative_wer(ground_truth, predicted[0])

        oracle_cer /= len(dataset)
        oracle_wer /= len(dataset)

        top1_cer /= len(dataset)
        top1_wer /= len(dataset)

        graph_results["top1_cer"].append(top1_cer)
        graph_results["top1_wer"].append(top1_wer)
        graph_results["oracle_cer"].append(oracle_cer)
        graph_results["oracle_wer"].append(oracle_wer)

    pd.DataFrame.from_dict(graph_results).to_csv("graph_results.csv", index=False)
