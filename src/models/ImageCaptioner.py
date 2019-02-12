# -*- coding: utf-8 -*-
"""
A PyTorch CNN-RNN model for Image Captioning.

Based on the Show & Tell Architecture
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class ImageCaptioner(nn.Module):
    """
    A PyTorch CNN-RNN model for Image Captioning.

    This class inherits from the torch.nn.Module class.
    """

    def __init__(self, im_embedding_size, embed_size,
                 hidden_size, vocab_size,
                 max_caption_length=35, start_id=1, end_id=2,
                 num_layers=1, dropout_prob=0.2,
                 rnn_type='lstm', rnn_dropout_prob=0.2):
        """
        Constructs the ImageCaptioner CNN-RNN

        Args:
            im_embedding_size: Size of the image embedding from the CNN
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
        super(ImageCaptioner, self).__init__()

        # Make class variables from input params
        self.im_embedding_size = im_embedding_size
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

        # Layers for preprocessing image embedding
        self.inp = nn.Linear(im_embedding_size, embed_size)
        self.inp_dropout = nn.Dropout(dropout_prob)
        self.inp_bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        # Layers for preprocessing captions
        self.embed = nn.Embedding(vocab_size, embed_size)

        # RNN
        self.rnn = rnn_type(embed_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=(0 if num_layers == 1
                                     else rnn_dropout_prob))

        # Linear layer on output
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, im_embeddings, caption_embeddings=None, mode='train'):
        """
        Compute the forward pass of the network

        Args:
            im_embeddings: Image embeddings from CNN
            caption_embeddings: Numerically encoded captions

        Returns:
            The network probability outputs
        """

        if mode == 'test':
            return self.predict(im_embeddings)

        # Prepare image/captions for passing into first RNN
        im_embeddings = self.inp(im_embeddings)
        im_embeddings = self.inp_dropout(im_embeddings)

        # Drop end tag
        caption_embeddings = self.embed(caption_embeddings[:, :-1])

        # Join the inputs sequentially
        inputs = torch.cat((im_embeddings, caption_embeddings), 1)

        # Pass through RNN
        outputs, _ = self.rnn(inputs)
        outputs = self.out(outputs)

        return outputs

    def predict(self, im_embeddings, return_probs=False,
                beam_size=None, desired_num_captions=1):
        """
        Predicts the captions for the given image embeddings


        Args:
            im_embeddings: Image embeddings from CNN
            return_probs: Option to return probabilities
            beam_size: Size of beam for beam search
            desired_num_captions: Top N captions to return

        Returns:
            The predicted captions, and optionally the output probabilities
        """

        # Prepare the image for passing through the first RNN
        if len(im_embeddings.size()) == 2:
            batch_size = 1
            im_embeddings = self.inp(im_embeddings.unsqueeze(0))
        else:
            batch_size, _, _ = im_embeddings.size()
            im_embeddings = self.inp(im_embeddings)

        # Compute caption using highest probability word recursively
        if not beam_size:
            # Pass image through network, and initialize storage tensors
            output, hidden = self.rnn(im_embeddings)

            captions = torch.zeros(batch_size, self.max_len)
            captions[:, 0] = self.start_id

            if return_probs:
                probs = torch.zeros(batch_size, self.max_len, self.output_size)
                output = self.out(output).squeeze(1)
                probs[:, 0, :] = output.cpu()

            word_embedding = captions[:, 0].unsqueeze(1).long()

            # Move tensors to GPU if available
            if torch.cuda.is_available():
                captions = captions.cuda()
                word_embedding = word_embedding.cuda()
                if return_probs:
                    probs = probs.cuda()

            # Recursively pass into RNN highest probability word
            for i in range(1, self.max_len - 1):
                word_embedding = self.embed(word_embedding)
                output, hidden = self.rnn(word_embedding, hidden)
                output = self.out(output).squeeze(1)

                # Store probability
                if return_probs:
                    probs[:, i, :] = output.cpu()

                if i < self.max_len - 1:
                    captions[:, i] = output.argmax(1)
                    word_embedding = output.argmax(1).unsqueeze(1)

                    # Break if all tags for current iteration are end tags
                    if not return_probs and np.all((
                                                    captions[:, i]
                                                    == self.end_id
                                                   ).cpu().numpy()):
                        break
        # Conduct beam search to find highest probable sentence
        else:
            # Initialize storage tensors and move to GPU
            captions = torch.zeros(
                batch_size, desired_num_captions, self.max_len)
            if return_probs:
                probs = torch.zeros(
                    batch_size, desired_num_captions,
                    self.max_len, self.output_size)

            if torch.cuda.is_available():
                captions = captions.cuda()
                if return_probs:
                    probs = probs.cuda()

            # Perform beam search for each image in batch
            for batch_id in range(batch_size):
                if return_probs:
                    captions[batch_id], probs[batch_id] = self.beam_search(
                        im_embeddings[batch_id].unsqueeze(0),
                        return_probs=return_probs,
                        beam_size=beam_size,
                        top_k=desired_num_captions)
                else:
                    captions[batch_id] = self.beam_search(
                        im_embeddings[batch_id].unsqueeze(0),
                        return_probs=return_probs,
                        beam_size=beam_size,
                        top_k=desired_num_captions)

            # Remove extra dimension
            if desired_num_captions == 1:
                captions = captions.squeeze(1)
                if return_probs:
                    probs = probs.squeeze(1)

        if return_probs:
            return captions, probs

        return captions

    def beam_search(self, output, hidden=None,
                    return_probs=False, beam_size=10, top_k=1):
        """
        Conducts beam search with the network

        Args:
            output: Input to RNN
            hidden: Hidden input to RNN
            return_probs: Option to return probabilities
            beam_size: Size of beam for beam search
            top_k: Top k captions to return

        Returns:
            The predicted captions, and optionally output probabilities
        """

        # Storage vector to store results
        if return_probs:
            idx_sequences = [[[], 0.0, output, hidden,
                              np.zeros((self.max_len,
                                        self.output_size))]]
        else:
            idx_sequences = [[[], 0.0, output, hidden]]

        for i in range(self.max_len):
            # Storage vector for top candidates
            candidates = []

            for idx_seq in idx_sequences:
                # Find outputs for current idx sequence
                output, hidden = self.rnn(idx_seq[2],
                                          idx_seq[3])
                outputs = self.out(output.squeeze(1))

                # Take softmax and find beam_size top results
                output_softmax = F.log_softmax(outputs, -1)
                top_probs, top_idx = output_softmax.topk(
                    beam_size, 1)
                top_idx = top_idx.squeeze(0)

                # Find best sentences for next round
                for j in range(beam_size):
                    next_idx_seq, log_prob = idx_seq[0][:], idx_seq[1]
                    next_idx_seq.append(top_idx[j].item())
                    log_prob += top_probs[0][j].item()

                    if return_probs:
                        idx_seq[4][i] = outputs.squeeze(0).detach().cpu().numpy()

                    output = self.embed(top_idx[j].unsqueeze(0)).unsqueeze(0)
                    # Store best
                    if return_probs:
                        candidates.append(
                            [next_idx_seq, log_prob, output,
                             hidden, idx_seq[4]])
                    else:
                        candidates.append(
                            [next_idx_seq, log_prob, output,
                             hidden])

            # Sort in order of probability
            candidates = sorted(candidates,
                                key=lambda x: x[1],
                                reverse=True)
            idx_sequences = candidates[:beam_size]

        if return_probs:
            return (torch.Tensor([idx_seq[0]
                                  for idx_seq in idx_sequences[:top_k]]),
                    torch.Tensor([idx_seq[4]
                                  for idx_seq in idx_sequences[:top_k]]))
        return torch.Tensor([idx_seq[0]
                             for idx_seq in idx_sequences[:top_k]])
