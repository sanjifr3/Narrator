"""
A PyTorch CNN-RNN model for Video Captioning

An re-implementation of the S2VT network.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class S2VTCaptioner(nn.Module):
    """
    A PyTorch CNN-RNN model for video captioning.

    This class inherits from the torch.nn.Module class
    """

    def __init__(self, vid_embedding_size, embed_size, hidden_size,
                 vocab_size, max_caption_length=35, start_id=1, end_id=2,
                 num_layers=1, dropout_prob=0.2, rnn_type='lstm',
                 rnn_dropout_prob=0.2):
        """
        Constructs the S2VT VideoCaptioner CNN-RNN

         Args:
            vid_embedding_size: Size of the vid embedding from the CNN
            embed_size: Word embedding size
            hidden_size: Hidden size of RNN
            vocab_size: Size of vocabulary
            max_caption_length: Maximum size of a caption
            start_id: Tag of starting word in vocabulary
            end_id: Tag of ending word in vocabulary
            num_layers: Number of layers for RNN
            dropout_prob: Probability of dropout for image input
            rnn_type: Type of RNN unit to use
            rnn_dropout_prob: Dropout probability for RNN
                              (only if num_layers>1)

        Returns:
            A PyTorch network model
        """
        super(S2VTCaptioner, self).__init__()

        # Make class variables from input params
        self.vid_embedding_size = vid_embedding_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.max_len = max_caption_length
        self.start_id = start_id
        self.end_id = end_id

        # Select rnn unit
        if rnn_type.lower() == 'lstm':
            rnn_type = nn.LSTM
        else:
            rnn_type = nn.GRU

        # Layers for preprocessing frame embeddings
        self.inp = nn.Linear(vid_embedding_size, hidden_size)
        self.inp_dropout = nn.Dropout(dropout_prob)

        # Layers for preprocessing captions
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Input and output RNNs
        self.inp_rnn = rnn_type(hidden_size, hidden_size, num_layers,
                                batch_first=True,
                                dropout=(0 if num_layers == 1
                                         else rnn_dropout_prob))

        self.out_rnn = rnn_type(hidden_size + embed_size,
                                hidden_size,
                                num_layers,
                                batch_first=True,
                                dropout=(0 if num_layers == 1
                                         else rnn_dropout_prob))

        # FC on output
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, vid_embeddings,
                caption_embeddings=None, mode='train'):
        """
        Compute the forward pass of the network

        Args:
            vid_embeddings: Video embeddings from the CNN
            caption_embeddings: Numerically encoded captions

        Returns:
            The network probability outputs
        """

        if caption_embeddings is None:
            mode = 'test'

        if mode == 'test':
            return self.predict(vid_embeddings)

        batch_size, num_frames, _ = vid_embeddings.size()

        # Pass video frames through linear layer and dropout
        vid_embeddings = self.inp(vid_embeddings)
        vid_embeddings = self.inp_dropout(vid_embeddings)

        # Pad videos at end to caption length
        padding_frames = Variable(
            vid_embeddings.data.new(
                batch_size,
                self.max_len - 1,
                self.hidden_size)).zero_()
        vid_embeddings = torch.cat((vid_embeddings, padding_frames), 1)

        # Pass padded video embeddings through input rnn
        self.inp_rnn.flatten_parameters()
        inp_output, _ = self.inp_rnn(vid_embeddings)

        # Pad captions in front by number of frames and embed
        padding_captions = Variable(
            caption_embeddings.data.new(
                batch_size, num_frames)).zero_()
        caption_embeddings = torch.cat(
            (padding_captions, caption_embeddings), 1)

        # drop last as either <pad> or <end>
        caption_embeddings = self.embed(caption_embeddings[:, :-1])

        # Merge output from input rnn and caption embeddings and
        # input into output rnn
        out_input = torch.cat((inp_output, caption_embeddings), 2)
        self.out_rnn.flatten_parameters()
        out_output, _ = self.out_rnn(out_input)

        # Pass caption related ouput through linear output layer
        # (don't care about the rest)
        out_output = self.out(out_output[:, -(self.max_len - 1):])
        out_output = F.log_softmax(out_output, dim=1)

        return out_output

    def predict(self, vid_embeddings, return_probs=False):
        """
        Predicts the captions for the given video embeddings

        Args:
            vid_embedding: Video embeddings from the CNN
            return_probs: Option to return probabilities
            desired_num_captions: Top N captions to return

        Returns:
            The predicted captions, and optionally the output probabilities
        """

        # Prepare video embeddings for passing through RNN
        if len(vid_embeddings.size()) == 2:
            batch_size = 1
            vid_embeddings = self.inp(vid_embeddings.unsqueeze(0))
        else:
            batch_size, num_frames, _ = vid_embeddings.size()
            vid_embeddings = self.inp(vid_embeddings)

        # Get padding captions
        padding_captions = Variable(
            vid_embeddings.data.new(
                batch_size, num_frames).zero_()).long()

        # Get a padding frame
        padding_frame = Variable(
            vid_embeddings.data.new(
                batch_size, 1, self.hidden_size)).zero_()

        # Move to gpu if available
        if torch.cuda.is_available():
            padding_captions = padding_captions.cuda()
            padding_frame = padding_frame.cuda()

        inp_hidden = None
        out_hidden = None

        # Embed captions
        padding_captions = self.embed(padding_captions)

        # Pass through videos + padding captions
        self.inp_rnn.flatten_parameters()
        inp_output, inp_hidden = self.inp_rnn(vid_embeddings,
                                              inp_hidden)

        out_input = torch.cat((inp_output, padding_captions), 2)

        self.out_rnn.flatten_parameters()
        _, out_hidden = self.out_rnn(out_input, out_hidden)

        # Create storage vectors
        captions = torch.zeros(batch_size, self.max_len)
        captions[:, 0] = self.start_id
        if torch.cuda.is_available():
            captions = captions.cuda()

        if return_probs:
            probs = torch.zeros(
                batch_size, self.max_len - 1, self.output_size)
            if torch.cuda.is_available():
                probs = probs.cuda()

        # Start with start tag
        word_embedding = Variable(torch.LongTensor(
            [self.start_id] * batch_size)).unsqueeze(1)

        if torch.cuda.is_available():
            word_embedding = word_embedding.cuda()

        # Compute captions using the highest probability word recursively
        for i in range(self.max_len - 1):
            self.inp_rnn.flatten_parameters()
            inp_output, inp_hidden = self.inp_rnn(
                padding_frame, inp_hidden)
            word_embedding = self.embed(word_embedding)
            out_input = torch.cat((inp_output, word_embedding), 2)

            self.out_rnn.flatten_parameters()
            out_output, out_hidden = self.out_rnn(out_input,
                                                  out_hidden)

            out_output = self.out(out_output.squeeze(1))
            out_output = F.log_softmax(out_output, dim=1)
            probs[:, i] = out_output

            _, captions[:, i + 1] = torch.max(out_output, 1)
            word_embedding = captions[:, i + 1].long().unsqueeze(1)

        return captions, probs
