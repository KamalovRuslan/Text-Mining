# Models for word alignment ibm model 2
import numpy as np

class TranslationModel:
    "Models conditional distribution over trg words given a src word, i.e. t(f|e)."

    def __init__(self, src_corpus, trg_corpus):

        self._src_trg_counts = {}
        self._trg_given_src_probs = {}

        src_vocab = set()
        trg_vocab = set()

        # model with p(f|e) = p(f, tag(f) | e, tag(e))
        #for i, src_sent in enumerate(src_corpus):
        #    for j, src in enumerate(src_sent):
        #        src_vocab.add(src + src_tags_corpus[i][j])
        #for i, trg_sent in enumerate(trg_corpus):
        #    for j, trg in enumerate(trg_sent):
        #        trg_vocab.add(trg+trg_tags_corpus[i][j])

        for src_sent in src_corpus:
            for src in src_sent:
                src_vocab.add(src)
        for trg_sent in trg_corpus:
            for trg in trg_sent:
                trg_vocab.add(trg)

        init_prob = 1. / len(trg_vocab)
        init_dist = {trg_plus_trg_tag: init_prob for trg_plus_trg_tag in trg_vocab}
        self._trg_given_src_probs = {src_plus_src_tag: init_dist for src_plus_src_tag in src_vocab}


        print "TranslationModel completed"

    def get_conditional_prob(self, src_token, trg_token):
        "Return the conditional probability of trg_token given src_token, i.e. t(f|e)."
        if src_token not in self._trg_given_src_probs:
            return 1.0
        if trg_token not in self._trg_given_src_probs[src_token]:
            return 1.0
        return self._trg_given_src_probs[src_token][trg_token]

    def collect_statistics(self, src_tokens, trg_tokens, posterior_matrix):
        "Accumulate counts of translations from matrix: matrix[j][i] = p(a_j=i|e, f)"
        assert len(posterior_matrix) == len(trg_tokens)
        for posterior in posterior_matrix:
            assert len(posterior) == len(src_tokens)
        # Hint - You just need to count how often each src and trg token are aligned
        # but since we don't have labeled data you'll use the posterior_matrix[j][i]
        # as the 'fractional' count for src_tokens[i] and trg_tokens[j].

        for i, src in enumerate(src_tokens):
            if not src in self._src_trg_counts:
                self._src_trg_counts[src] = {}
            for j, trg in enumerate(trg_tokens):
                if not trg in self._src_trg_counts[src]:
                    self._src_trg_counts[src][trg] = posterior_matrix[j][i]
                else:
                    self._src_trg_counts[src][trg] += posterior_matrix[j][i]

        # model with p(f|e) = p(f, tag(f) | e, tag(e))
        #for i, s_tag in enumerate(src_tags):
        #    src = src_tokens[i] + s_tag
        #    if not src in self._src_trg_counts:
        #        self._src_trg_counts[src] = {}
        #    for j, t_tag in enumerate(trg_tags):
        #        trg = trg_tokens[j] + t_tag
        #        if not trg in self._src_trg_counts[src]:
        #            self._src_trg_counts[src][trg] = posterior_matrix[j][i]
        #        else:
        #            self._src_trg_counts[src][trg] += posterior_matrix[j][i]

        # assert False, "Collect statistics here!"

    def recompute_parameters(self):
        "Reestimate parameters and reset counters."
        # Hint - Just normalize the self._src_and_trg_counts so that the conditional
        # distributions self._trg_given_src_probs are correctly normalized to give t(f|e).

        self._trg_given_src_probs = {}
        for src in self._src_trg_counts:
            dist = self._src_trg_counts[src]
            src_sum = sum(dist.values())
            self._trg_given_src_probs[src] = {trg: float(dist[trg]) / src_sum for trg in dist}
        self._src_trg_counts = {}

class PriorModel:
    "Models the prior probability of an alignment given only the sentence lengths and token indices."

    def __init__(self, src_corpus, trg_corpus):
        "Add counters and parameters here for more sophisticated models."
        self._distance_counts = {}
        self._distance_probs = {}

        src_length = set()
        trg_length = set()

        for src_sent in src_corpus:
            src_length.add(len(src_sent))
        for trg_sent in trg_corpus:
            trg_length.add(len(trg_sent))

        for s_length in src_length:
            self._distance_probs[s_length] = {}
            for t_length in trg_length:
                self._distance_probs[s_length][t_length] = {}
                for j in range(t_length):
                    init_prob = 1. / s_length
                    self._distance_probs[s_length][t_length][j] = {i: init_prob for i in range(s_length)}

        print "PriorModel completed"


    def get_prior_prob(self, src_index, trg_index, src_length, trg_length):
        "Returns a uniform prior probability."
        # Hint - you can probably improve on this, but this works as is.
        #return 1. / src_length
        return self._distance_probs[src_length][trg_length][trg_index][src_index]

    def collect_statistics(self, src_length, trg_length, posterior_matrix):
        "Count the necessary statistics from this matrix if needed."
        if not src_length in self._distance_counts:
            self._distance_counts[src_length] = {}
        if not trg_length in self._distance_counts[src_length]:
            self._distance_counts[src_length][trg_length] = {}
        for j in range(trg_length):
            if not j in self._distance_counts[src_length][trg_length]:
                self._distance_counts[src_length][trg_length][j] = {}
            for i in range(src_length):
                if not i in self._distance_counts[src_length][trg_length][j]:
                    self._distance_counts[src_length][trg_length][j][i] = posterior_matrix[j][i]
                else:
                    self._distance_counts[src_length][trg_length][j][i] += posterior_matrix[j][i]
        #pass

    def recompute_parameters(self):
        "Reestimate the parameters and reset counters."
        self._distance_probs = {}
        for src_length in self._distance_counts:
            if not src_length in self._distance_probs:
                self._distance_probs[src_length] = {}
            for trg_length in self._distance_counts[src_length]:
                if not trg_length in self._distance_probs[src_length]:
                    self._distance_probs[src_length][trg_length] = {}
                for j in self._distance_counts[src_length][trg_length]:
                    sum_j = sum(self._distance_counts[src_length][trg_length][j].values())
                    self._distance_probs[src_length][trg_length][j] = {i: self._distance_counts[src_length][trg_length][j][i] / sum_j for i in self._distance_counts[src_length][trg_length][j].keys()}

        #self._distance_counts = {}
        #pass
