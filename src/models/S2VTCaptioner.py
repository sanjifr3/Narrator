import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import random
from torch.autograd import Variable

class S2VTCaptioner(nn.Module):
  def __init__(self, vid_embedding_size, embed_size, hidden_size, vocab_size,
        max_caption_length, start_id=1, end_id=2, num_layers=1, dropout_prob=0.2, 
        rnn_type='lstm',rnn_dropout_prob=0.2):
    super(S2VTCaptioner, self).__init__()

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

    self.inp = nn.Linear(vid_embedding_size, hidden_size) # hidden_size
    self.inp_dropout = nn.Dropout(dropout_prob)

    self.inp_rnn = rnn_type(hidden_size, hidden_size, num_layers, 
                              batch_first=True, 
                              dropout=(0 if num_layers == 1 else rnn_dropout_prob))

    self.embed = nn.Embedding(vocab_size, embed_size)

    self.out_rnn = rnn_type(hidden_size + embed_size, hidden_size, num_layers,
                              batch_first=True,
                              dropout=(0 if num_layers == 1 else rnn_dropout_prob))

    self.out = nn.Linear(hidden_size, vocab_size)

  def forward(self, vid_embeddings, caption_embeddings=None, mode='train'):
    if caption_embeddings is None:
        mode = 'test'
       
    batch_size, num_frames,_ = vid_embeddings.size()

    # Pass video frames through linear layer and dropout
    vid_embeddings = self.inp(vid_embeddings)
    vid_embeddings = self.inp_dropout(vid_embeddings)

    # Pad videos at end to caption length
    padding_frames = Variable(vid_embeddings.data.new(batch_size, self.max_len-1, self.hidden_size)).zero_()
    vid_embeddings = torch.cat((vid_embeddings, padding_frames), 1)

    # Pass padded video embeddings through input rnn
    self.inp_rnn.flatten_parameters()
    inp_output,_ = self.inp_rnn(vid_embeddings)

    # Pad captions in front by number of frames and embed
    padding_captions = Variable(caption_embeddings.data.new(batch_size, num_frames)).zero_()
    caption_embeddings = torch.cat((padding_captions, caption_embeddings), 1) 
    caption_embeddings = self.embed(caption_embeddings[:,:-1]) # drop last as either <pad> or <end> 

    # Merge output from input rnn and caption embeddings and input into output rnn
    out_input = torch.cat((inp_output, caption_embeddings),2)
    self.out_rnn.flatten_parameters()
    out_input,_ = self.out_rnn(out_input)

    # Pass caption related ouput through linear output layer (don't care about the rest)
    out_output = self.out(out_input[:,-(self.max_len-1):])
    out_output = F.log_softmax(out_output, dim=1)

    return out_output

  def predict(self, vid_embeddings, return_probs=False, beam_size=None, desired_num_captions=1):
    if len(vid_embeddings.size()) == 2:
      batch_size = 1
      vid_embeddings = self.inp(vid_embeddings.unsqueeze(0))
    else:
      batch_size, num_frames, _ = vid_embeddings.size()
      vid_embeddings = self.inp(vid_embeddings)

    if not beam_size:
      padding_captions = Variable(vid_embeddings.data.new(batch_size, num_frames).zero_()).long()
      padding_frame = Variable(vid_embeddings.data.new(batch_size, 1, self.hidden_size)).zero_()

      if torch.cuda.is_available():
        padding_captions = padding_captions.cuda()
        padding_frame = padding_frame.cuda()

      inp_hidden = None
      out_hidden = None

      #self.inp_rnn.flatten_parameters()
      #self.out_rnn.flatten_parameters()

      padding_captions = self.embed(padding_captions)

      inp_output, inp_hidden = self.inp_rnn(vid_embeddings, inp_hidden)
      out_input = torch.cat((inp_output, padding_captions),2)
      out_output, out_hidden = self.out_rnn(out_input, out_hidden)

      captions = torch.zeros(batch_size, self.max_len)
      captions[:,0] = self.start_id
      if torch.cuda.is_available():
        captions = captions.cuda()

      if return_probs:
        probs = torch.zeros(batch_size, self.max_len-1, self.output_size)
        if torch.cuda.is_available():
          probs = probs.cuda()

      word_embedding = Variable(torch.LongTensor([self.start_id] * batch_size)).unsqueeze(1)

      if torch.cuda.is_available():
        word_embedding = word_embedding.cuda()

      #print (inp_hidden)

      for i in range(self.max_len - 1):
        #self.inp_rnn.flatten_parameters()
        #self.out_rnn.flatten_parameters()
        inp_output, inp_hidden = self.inp_rnn(padding_frame, inp_hidden)

        word_embedding = self.embed(word_embedding)
        out_input = torch.cat((inp_output, word_embedding), 2)

        out_output, out_hidden = self.out_rnn(out_input, out_hidden)

        out_output = self.out(out_output.squeeze(1))
        out_output = F.log_softmax(out_output, dim=1)
        probs[:,i] = out_output
 
        _, captions[:,i+1] = torch.max(out_output,1)
        word_embedding = captions[:,i+1].long().unsqueeze(1)

      return captions, probs

















      inp_output,inp_hidden = self.inp_rnn(vid_embeddings)

      input2 = torch.cat((inp_output,padding_captions),2)
      self.out_rnn.flatten_parameters()
      out_output, out_hidden = self.out_rnn(input2)

      captions = torch.zeros(batch_size, self.max_len)
      
      if torch.cuda.is_available():
        captions = captions.cuda()

      if return_probs:
        probs = torch.zeros(batch_size, self.max_len, self.output_size)
        if torch.cuda.is_available():
          probs = probs.cuda()
        probs[:,0] = self.out(out_output[:,-1]).squeeze(1) # Pass only final frame through (should be max at start_id)

      captions[:,0] = self.start_id
      word_embedding = captions[:,0].unsqueeze(1).long()

      #padding_frame = torch.zeros(batch_size, 1, self.hidden_size).cuda()
      for i in range(1, self.max_len):
        inp_output, inp_hidden = self.inp_rnn(padding_frame, inp_hidden)

        word_embedding = self.embed(word_embedding)
        input2 = torch.cat((inp_output, word_embedding),2)

        out_output, out_hidden = self.out_rnn(input2, out_hidden)
        out_output = self.out(out_output).squeeze(1)

        if return_probs:
          probs[:,i] = out_output

        if i < self.max_len - 1:
          captions[:,i] = out_output.argmax(1)
          word_embedding = captions[:,i].unsqueeze(1).long()

          if not return_probs and np.all((captions[:,i] == self.end_id).cpu().numpy()):
            break
    else:
      # Beam size implementation
      pass

    if return_probs:
      return captions, probs
    else:
      return captions




      
      word_embedding = self.embed(word_embedding)













    
    if not beam_size:
      pass





    vid_embeddings = self.inp(vid_embeddings)
    vid_embeddings = self.inp_dropout(vid_embeddings)
    #vid_embeddings = self.inp_bn(vid_embeddings)

    caption_embeddings = self.embed(caption_embeddings[:,:-1])

    state1 = None
    state2 = None

    if mode == 'simple_train':
      output1, state1 = self.inp_rnn(vid_embeddings, state1)
      caption_embeddings = self.embed(caption_embeddings[:,:-1])
      print (caption_embeddings.size())
      print (output1.size())
      input2 = torch.cat((output1, caption_embeddings.unsqueeze(1)), dim=2)
      outputs2, state2 = self.out_rnn(input2,state2)
      outputs2 = self.out(outputs2)

      return outputs2

    padding_words = Variable(vid_embeddings.data.new(batch_size, num_frames, self.embed_size)).zero_()
    padding_frames = Variable(vid_embeddings.data.new(batch_size, 1, self.hidden_size)).zero_()
    
    output1, state1 = self.inp_rnn(vid_embeddings, state1)
    input2 = torch.cat((output1, padding_words), dim=2)
    output2, state2 = self.out_rnn(input2, state2)

    all_preds = []
    all_probs = []

    if mode == 'train':
      for i in range(self.max_len):
        current_words = self.embed(caption_embeddings[:,i])
        self.inp_rnn.flatten_parameters()
        self.out_rnn.flatten_parameters()

        output1, state1 = self.inp_rnn(padding_frames, state1)
        input2 = torch.cat((output1, current_words.unsqueeze(1)), dim=2)
        output2, state2 = self.out_rnn(input2, state2)
        output2 = self.out(output2.squeeze(1))
        output2 = self.out_dropout(output2)

        # Add #
        all_probs.append(output2.unsqueeze(1))

        # Remove # 
        #logits = F.log_softmax(logits, dim=1)
        #all_probs.append(logits.unsqueeze(1))
      all_probs = torch.cat(all_probs,1)
      #print (all_probs[12])
    else:
      current_words = self.embed(
         Variable(torch.LongTensor([self.start_id] * batch_size)).cuda())
      
      for i in range(self.max_len):
        self.inp_rnn.flatten_parameters()
        self.out_rnn.flatten_parameters()
       
        output1, state1 = self.inp_rnn(padding_frames, state1)
        input2 = torch.cat((output1, current_words.unsqueeze(1)), dim=2)
        output2, state2 = self.out_rnn(input2, state2)
        output2 = self.out(output2.squeeze(1))
        output2 = self.out_dropout(output2)

        # Add # 
        preds = output2.argmax(1)
        current_words = self.embed(preds)
        all_preds.append(preds.unsqueeze(1))
        all_probs.append(output2.unsqueeze(1))

        # Remove # 
        #logits = F.log_softmax(logits, dim=1)
        #all_probs.append(logits.unsqueeze(1))
        #_, preds = torch.max(logits, 1)
        #current_words = self.embed(preds)
        #all_preds.append(preds.unsqueeze(1))

      all_probs = torch.cat(all_probs, 1)
      all_preds = torch.cat(all_preds, 1)

      #print( all_preds[12])
      #print (all_probs[12])
    return all_probs, all_preds