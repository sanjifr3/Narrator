import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import random
from torch.autograd import Variable
import numpy as np

class VideoCaptioner(nn.Module):
  def __init__(self, vid_embedding_size, embed_size, hidden_size, vocab_size,
        max_caption_length=35, start_id=1, end_id=2, num_layers=1, dropout_prob=0.2, 
        rnn_type='lstm',rnn_dropout_prob=0.2):
    super(VideoCaptioner, self).__init__()

    self.vid_embedding_size = vid_embedding_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.output_size = vocab_size
    self.max_len = max_caption_length

    self.start_id = start_id
    self.end_id = end_id

    if rnn_type.lower() == 'lstm':
      rnn_type = nn.LSTM 
    else:
      rnn_type = nn.GRU

    # Preparing image for LSTM
    self.inp = nn.Linear(vid_embedding_size, embed_size)
    self.inp_dropout = nn.Dropout(dropout_prob)
    self.inp_bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    # Preparing caption for LSTM
    self.embed = nn.Embedding(vocab_size, embed_size)

    # LSTM
    self.rnn = rnn_type(embed_size, hidden_size, num_layers, 
                          batch_first=True, 
                          dropout=(0 if num_layers == 1 else rnn_dropout_prob))

    self.out = nn.Linear(hidden_size, vocab_size)

  def forward(self, vid_embeddings, caption_embeddings=None, mode='train'):

    if mode=='test':
      return predict(vid_embeddings)

    batch_size, num_frames,_ = vid_embeddings.size()
 
    # Prepare videos for passing into first RNN
    vid_embeddings = self.inp(vid_embeddings)
    vid_embeddings = self.inp_dropout(vid_embeddings)

    # Drop last as no point inputting <PAD> or <EOS>
    caption_embeddings = self.embed(caption_embeddings[:,:-1])

    # Concatenate image embedding and word embeddings
    inputs = torch.cat((vid_embeddings, caption_embeddings),1)

    # Pass all through RNNs   
    outputs,_ = self.rnn(inputs)
    
    # Pass through linear output
    outputs = self.out(outputs[:,-self.max_len:])
    
    return outputs

  def predict(self, vid_embeddings, return_probs=False, beam_size=None, desired_num_captions=1):
    if len(vid_embeddings.size()) == 2:
      batch_size = 1
      vid_embeddings = self.inp(vid_embeddings.unsqueeze(0))
    else:
      batch_size,_,_ = vid_embeddings.size()
      vid_embeddings = self.inp(vid_embeddings)

    if not beam_size:
      output, hidden = self.rnn(vid_embeddings)

      captions = torch.zeros(batch_size, self.max_len)
      captions[:,0] = self.start_id 
      if return_probs:
        probs = torch.zeros(batch_size, self.max_len, self.output_size)
        probs[:,0] = self.out(output[:,-1]).squeeze(1).cpu()
        #output = self.out(output).squeeze(1)
        #probs[:,0] = output[:,-1].cpu()

      word_embedding = captions[:,0].unsqueeze(1).long()

      if torch.cuda.is_available():
        word_embedding = word_embedding.cuda()
        captions = captions.cuda()
        if return_probs:
          probs = probs.cuda()

      for i in range(1,self.max_len):
        word_embedding = self.embed(word_embedding)
        output,hidden = self.rnn(word_embedding,hidden)
        output = self.out(output).squeeze(1)

        # Store probability
        if return_probs:
          probs[:,i] = output.cpu()

        if i < self.max_len - 1:
          captions[:,i] = output.argmax(1)
          word_embedding = output.argmax(1).unsqueeze(1)

          if not return_probs and np.all((captions[:,i] == self.end_id).cpu().numpy()):
            break

    else:
      output, hidden = self.rnn(vid_embeddings[:,:-1])
      captions = torch.zeros(batch_size, desired_num_captions, self.max_len)
      if return_probs:
        probs = torch.zeros(batch_size, desired_num_captions, self.max_len, self.output_size)

      if torch.cuda.is_available():
        captions = captions.cuda()
        if return_probs:
          probs = probs.cuda()

      for batch_id in range(batch_size):
        if return_probs:
          captions[batch_id], probs[batch_id] = self.beamSearch(vid_embeddings[:,-1][batch_id].unsqueeze(0).unsqueeze(0), 
                                                             (hidden[0][:,batch_id].unsqueeze(0),hidden[1][:,batch_id].unsqueeze(0)),
                                                              return_probs=return_probs,
                                                              beam_size=beam_size,
                                                              topK=desired_num_captions)
        else:
          captions[batch_id] = self.beamSearch(vid_embeddings[:,-1][batch_id].unsqueeze(0).unsqueeze(0), 
                                                 (hidden[0][:,batch_id].unsqueeze(0),hidden[1][:,batch_id].unsqueeze(0)),
                                                  return_probs=return_probs,
                                                  beam_size=beam_size,
                                                  topK=desired_num_captions)
      if desired_num_captions == 1:
        captions = captions.squeeze(1)
        if return_probs:
          probs = probs.squeeze(1)
      
    if return_probs:
      return captions, probs
    else:
      return captions


  def beamSearch(self, output, hidden=None, return_probs=False, beam_size=10, topK=1):   
      # Storage vector to store results
      if return_probs:
        idx_sequences = [[[], 0.0, output, hidden, np.zeros((self.max_len,self.output_size))]]
      else:
        idx_sequences = [[[], 0.0, output, hidden]]

      for i in range(self.max_len):
          candidates = [] # Storage vector for top candidates
          for idx_seq in idx_sequences:
              # Find outputs for current idx sequence
              output, hidden = self.rnn(idx_seq[2], idx_seq[3])
              outputs = self.out(output.squeeze(1))

              # Take softmax and find beam_size top results
              output_softmax = F.log_softmax(outputs, -1)
              top_probs, top_idx = output_softmax.topk(beam_size, 1)
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
                    candidates.append([next_idx_seq, log_prob, output, hidden, idx_seq[4]])
                  else:
                    candidates.append([next_idx_seq, log_prob, output, hidden])

          # Sort in order of probability
          candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
          idx_sequences = candidates[:beam_size]

      if return_probs:
        return torch.Tensor([idx_seq[0] for idx_seq in idx_sequences[:topK]]), torch.Tensor([idx_seq[4] for idx_seq in idx_sequences[:topK]])
      else:
        return torch.Tensor([idx_seq[0] for idx_seq in idx_sequences[:topK]])