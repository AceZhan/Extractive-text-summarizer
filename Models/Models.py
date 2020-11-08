import torch
import torch.nn as nn


class SentenceEncoderRNN(nn.Module):
    """ Encoder to create sentential representations 
     
    Encoder has 2 layers:
        - embedding layer
        - GRU layer
    """

    def __init__(self, vocab_size: int, input_size: int, hidden_size: int, layers: int = 1):
        """
        Parameters:
            vocab_size: number of words in vocab
            input_size: size of word embedding vectors i.e number of features each input has
            hidden_size: size of output sentence vector. i.e number of features the output will have
            layers: number of stacked RNN layers, default 1
        """
        super(SentenceEncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, input_size)

        # GRU layer
        self.rnn = nn.GRU(hidden_size, hidden_size)


    def forward(self, input, hidden):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data.
        """
        #input has shape (seq_length, batch_size)
        
        embedded = self.embedding(input).view(1, 1, -1)

        print(embedded.shape)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden


    def initHidden(self, device):
        "Initializes "
        return torch.zeros(1, 1, self.hidden_size, device=device)
