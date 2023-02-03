from collections import defaultdict
from dataclasses import dataclass

import torch
import numpy as np

np.seterr(divide="ignore")


@dataclass
class Stats:
    prob_total: np.float32 = -np.inf
    prob_text: np.float32 = 1.0
    prob_non_blank: np.float32 = -np.inf
    prob_blank: np.float32 = -np.inf
    lm_applied: bool = False


class CTCDecoder:
    def __init__(self, vocab_dict, lm):
        self.vocab = vocab_dict
        self.lm = lm

        # Id специальных токенов в словаре
        self.blank_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.word_sep_id = 4
        # word_sep_id должен быть заменен на пробел при декодировании
        # и не забудьте удалить пробелы в конце строки!

    def __apply_vocab(self, sequence):
        return " ".join(
            "".join(
                map(
                    lambda x: self.vocab[x] if x != self.word_sep_id else " ",
                    filter(
                        lambda x: x not in (self.blank_id, self.bos_id, self.unk_id),
                        sequence
                        if self.eos_id not in sequence
                        else sequence[: sequence.index(self.eos_id)],
                    ),
                )
            ).split()
        )

    def __apply_lm(self, beams, parent_beam, child_beam) -> None:
        """Calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars."""
        if not self.lm or beams[child_beam].lm_applied:
            return

        # take bigram if beam length at least 2
        if len(child_beam) > 1:
            c = self.vocab[child_beam[-2]]
            d = self.vocab[child_beam[-1]]
            ngram_prob = self.lm.get_char_bigram(c, d)
        # otherwise take unigram
        else:
            c = self.vocab[child_beam[-1]]
            ngram_prob = self.lm.get_char_unigram(c)

        lm_factor = 0.01  # influence of language model
        beams[child_beam].prob_text = beams[parent_beam].prob_text + lm_factor * np.log(
            ngram_prob
        )  # probability of char sequence
        beams[child_beam].lm_applied = True  # only apply LM once per beam entry

    def __get_best_beams(self, beams, beam_size):
        return sorted(
            beams.items(), key=lambda x: x[1].prob_total + x[1].prob_text, reverse=True
        )[:beam_size]

    def argmax_decode(self, ctc_logits: torch.tensor) -> str:
        """
        ctc_logits - ctc-матрица логитов размерности [TIME, VOCAB]
        """
        indices = ctc_logits.argmax(axis=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        result = [t.item() for t in indices]
        return self.__apply_vocab(result)

    def beam_search_decode(self, ctc_logits: torch.tensor, beam_size: int = 16):
        """
        ctc_logits - ctc-матрица логитов размерности [TIME, VOCAB]
        beam_size - размер бима(луча)
        """
        ctc_probs = ctc_logits.softmax(dim=-1)
        max_T, max_C = ctc_probs.shape

        ctc_probs = ctc_probs.log().numpy()

        prev = defaultdict(Stats)
        beam = ()
        prev[beam] = Stats()
        prev[beam].prob_blank = 0.0
        prev[beam].prob_total = 0.0

        # для каждого шага предсказания
        for t in range(max_T):
            curr = defaultdict(Stats)

            best_beams = self.__get_best_beams(prev, beam_size)

            # для каждого имеющегося бима
            for beam, _ in best_beams:

                prob_non_blank = -np.inf

                if beam:
                    prob_non_blank = prev[beam].prob_non_blank + ctc_probs[t, beam[-1]]

                prob_blank = prev[beam].prob_total + ctc_probs[t, 0]

                curr[beam].prob_non_blank = np.logaddexp(
                    curr[beam].prob_non_blank, prob_non_blank
                )
                curr[beam].prob_blank = np.logaddexp(curr[beam].prob_blank, prob_blank)
                curr[beam].prob_total = np.logaddexp(
                    curr[beam].prob_total, np.logaddexp(prob_blank, prob_non_blank)
                )

                curr[beam].prob_text = prev[beam].prob_text
                curr[beam].lm_applied = True

                # без пустого символа т.к. с ним всё сделали
                for c in range(1, max_C):
                    new_beam = beam + (c,)

                    if beam and beam[-1] == c:
                        prob_non_blank = prev[beam].prob_blank + ctc_probs[t, c]
                    else:
                        prob_non_blank = prev[beam].prob_total + ctc_probs[t, c]

                    curr[new_beam].prob_non_blank = np.logaddexp(
                        curr[new_beam].prob_non_blank, prob_non_blank
                    )
                    curr[new_beam].prob_total = np.logaddexp(
                        curr[new_beam].prob_total, prob_non_blank
                    )

                    self.__apply_lm(curr, beam, new_beam)

            prev = curr

        res = []
        for beam, _ in self.__get_best_beams(prev, beam_size):
            res.append(self.__apply_vocab(beam))

        return res
