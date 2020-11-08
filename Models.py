import torch
import torch.nn as nn


class SentenceEncoderRNN(nn.Module):
    """ Encoder to create sentential representations 
     
    Encoder has 2 layers:
        - embedding layer
        - GRU layer
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float, layers: int = 1):
        """
        Parameters:
            input_size: size of word embedding vectors i.e number of features each input has
            hidden_size: size of output sentence vector. i.e number of features the output will have
            layers: number of stacked RNN layers, default 1
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data.
        """
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
