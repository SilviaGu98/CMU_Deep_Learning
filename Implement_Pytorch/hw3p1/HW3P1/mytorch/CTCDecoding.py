import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set
        


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
        # symbol_dict = {i: self.symbol_set[i] for i in range(len(self.symbol_set))}
        
        for s in range(len(y_probs[0])):
            max_prob, max_prob_idx = 0,0

            for p in range(len(y_probs)):
                if y_probs[p][s][0] > max_prob:
                    max_prob = y_probs[p][s][0]
                    max_prob_idx = p

            path_prob *= max_prob
            if max_prob_idx == 0:
                decoded_path.append('-')
            else:
                decoded_path.append(self.symbol_set[max_prob_idx-1])
        # compress sequence
        compressed_decoded_path = ['']
        for str in decoded_path:
            if str == '-' or str == compressed_decoded_path[-1]:
                pass
            else:
                compressed_decoded_path.append(str)
        decoded_path = ''.join(compressed_decoded_path)

        return decoded_path, path_prob
        


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        bestPath, FinalPathScore = None, None
        
        
        #return bestPath, FinalPathScore
        raise NotImplementedError

    def initialize_paths(self, symbols, y_probs):
        path = '-'
        terminal_blank_paths = set()
        terminal_blank_path_scores = {}

        terminal_blank_paths.add(path)
        terminal_blank_path_scores[path] = y_probs[0]

        terminal_symbol_paths = set()
        terminal_symbol_path_scores = {}

        for i in range(len(symbols)):
            path = symbols[i]
            terminal_symbol_paths.add(path)
            terminal_symbol_path_scores[path] = y_probs[i+1]
        
        return terminal_blank_paths, terminal_blank_path_scores, terminal_symbol_paths, terminal_symbol_path_scores


