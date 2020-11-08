import torch

import Models as m

def test_sentence_encoder_inout_format():
   """ Tests that the Sentence encoder's input and output is working as expected
   """

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   VOCAB_SIZE = 100000     # VOCAB SIZE
   INPUT_SIZE = 50         # input (word) vector size
   OUTPUT_SIZE = 512       # expected output size

   # Create random input Tensor of share (1, 1, INPUT_SIZE)
   x = torch.randint(high=10, size=(INPUT_SIZE, 1))

   encoder = m.SentenceEncoderRNN(INPUT_SIZE, OUTPUT_SIZE)

   # initial hidden vector for first input
   encoder_hidden = encoder.initHidden(device=device) 
   # print(encoder_hidden)

   # for i in range(3):
   output, encoder_hidden = encoder(x, encoder_hidden)
   print(output)
   # print(output)

def main():
   test_sentence_encoder_inout_format()

if __name__ == "__main__":
    main()

