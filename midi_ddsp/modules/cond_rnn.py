#  Copyright 2022 The MIDI-DDSP Authors.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Abstract model for conditional autoregressive RNN used in MIDI-DDSP."""

import tensorflow as tf
from tqdm.autonotebook import tqdm

tfk = tf.keras
tfkl = tfk.layers


class StackedRNN(tfk.Model):
  """Stacked RNN implementated using tfkl.Layers."""

  def __init__(self, nhid=256, nlayers=2, rnn_type='gru', dropout=0.5):
    super().__init__()
    if rnn_type == 'gru':
      rnn_layer = tfkl.GRU
    elif rnn_type == 'lstm':
      rnn_layer = tfkl.LSTM
    else:
      raise TypeError('Unknown rnn_type')
    self.nhid = nhid
    self.nlayers = nlayers
    self.net = {
      str(i): rnn_layer(nhid, return_sequences=True, return_state=True, dropout=dropout) for i in range(nlayers)
    }

  def call(self, x, initial_state=None, training=False):
    states_out_all = []
    z_out = x
    if initial_state is None:
      initial_state = [None for _ in range(self.nlayers)]
    for i in range(self.nlayers):
      o = self.net[str(i)](z_out,
                           initial_state=initial_state[i],
                           training=training)

      z_out, states_out = self.net[str(i)](z_out,
                                           initial_state=initial_state[i],
                                           training=training)
      states_out_all.append(states_out)
    return z_out, states_out_all


class TwoLayerCondAutoregRNN(tfk.Model):
  """Conditional two-layer autoregressive RNN.
  The RNN here takes input from a conditioning sequence and its previous output.
  The RNN is trained using teacher forcing and inference in autoregressive mode.
  """

  def __init__(self, nhid, n_out, input_dropout=True, input_dropout_p=0.5,
               dropout=0.5, rnn_type='gru'):
    """Constructor."""
    super().__init__()
    self.n_out = n_out
    self.nhid = nhid
    self.input_dropout = input_dropout
    self.input_dropout_p = input_dropout_p
    if rnn_type == 'gru':
      self.rnn1 = tfkl.GRU(nhid, return_sequences=True, return_state=True,
                           dropout=dropout)
      self.rnn2 = tfkl.GRU(nhid, return_sequences=True, return_state=True,
                           dropout=dropout)
    elif rnn_type == 'lstm':
      self.rnn1 = tfkl.LSTM(nhid, return_sequences=True, return_state=True,
                            dropout=dropout)
      self.rnn2 = tfkl.LSTM(nhid, return_sequences=True, return_state=True,
                            dropout=dropout)
    else:
      raise ValueError('Unknown RNN type.')

  def _one_step(self, curr_cond, prev_out, prev_states, training=False):
    """One step inference."""
    print(f"TwoLayerCondAutoregRNN._one_step()")

    prev_states_1, prev_states_2 = prev_states
    curr_z_in = tf.concat([curr_cond, prev_out], -1)  # [batch_size, 1, dim]
    curr_z_in = self.encode_z(curr_z_in, training=training)
    curr_z_out, curr_states_1 = self.rnn1(curr_z_in,
                                          initial_state=prev_states_1,
                                          training=training)
    curr_z_out, curr_states_2 = self.rnn2(curr_z_out,
                                          initial_state=prev_states_2,
                                          training=training)
    curr_out = self.decode_out(curr_z_out)

    print(f"end of TwoLayerCondAutoregRNN._one_step()")

    return curr_out, (curr_states_1, curr_states_2)

  @tf.function()
  def autoregressive(self, cond, training=False, display_progressbar=False):
    """Autoregressive inference."""
    print(f"MidiExpreToF0AutoregDecoder.autoregressive()")
    cond_encode = self.encode_cond(cond)
    cond_encode_shape = tf.shape(cond_encode)
    batch_size = cond_encode_shape[0]
    length = cond_encode_shape[1]
    prev_out = tf.tile([[0.0]], [batch_size, self.n_out])[:, tf.newaxis, :]  # go_frame

    state_shape = [batch_size, self.nhid]
    initial_state = tf.zeros(state_shape, dtype=tf.float32)
    prev_states = (initial_state, initial_state)

    overall_outputs_shapes = {"f0_midi_dv_logits": (3, 1, 201), "f0_midi_dv_onehot": (3, 1, 201),
                              "curr_out": (3, 1, 201), }
    overall_outputs_per_key = {key: tf.TensorArray(dtype=cond_encode.dtype, size=length)
                               for key, _ in overall_outputs_shapes.items()}

    curr_out = {
      k: tf.zeros((batch_size, 1, v.shape[-1]), dtype=tf.float32)
      for k, v in self.sample_out(self.encode_out(prev_out, training=training), training=training).items()
    }

    print(f"MidiExpreToF0AutoregDecoder.autoregressive(): iterating, len={length}")
    print(f"type(length)={type(length)}")
    print(f"length={length}")
    print(f"tf.shape(length)={tf.shape(length)}")

    print(f"tf.range(length)={tf.range(length)}")

    def loop_body(i, prev_out, prev_states, overall_outputs_per_key):
      curr_cond = cond_encode[:, i, :][:, tf.newaxis, :]
      prev_out = self.encode_out(prev_out, training=training)
      curr_out, curr_states = self._one_step(curr_cond, prev_out, prev_states, training=training)
      curr_out = self.sample_out(curr_out, training=training)

      for key, tensor in curr_out.items():
        overall_outputs_per_key[key] = overall_outputs_per_key[key].write(i, tensor)

      return i + 1, curr_out['curr_out'], curr_states, overall_outputs_per_key

    _, _, _, overall_outputs_per_key = tf.while_loop(
      cond=lambda i, *_: i < length,
      body=loop_body,
      loop_vars=(0, prev_out, prev_states, overall_outputs_per_key)
    )

    if False:
      for i in tf.range(length):
        # print(f"MidiExpreToF0AutoregDecoder.autoregressive(): iterating, i={i}/{length}")

        curr_cond = cond_encode[:, i, :][:, tf.newaxis, :]
        prev_out = self.encode_out(prev_out, training=training)
        curr_out, curr_states = self._one_step(curr_cond, prev_out, prev_states, training=training)
        curr_out = self.sample_out(curr_out, training=training)

        print(f"curr_out={curr_out}")
        for key, tensor in curr_out.items():
          overall_outputs_per_key[key] = overall_outputs_per_key[key].write(i, tensor)

        prev_out, prev_states = curr_out['curr_out'], curr_states

    @tf.function
    def stack_and_concat(tensor_array, axis=1):
      i = tf.constant(0)
      length = tensor_array.size()
      output = tensor_array.read(0)

      def body(i, output):
        output = tf.concat([output, tensor_array.read(i + 1)], axis=axis)
        return i + 1, output

      _, output = tf.while_loop(
        cond=lambda i, *_: i + 1 < length,
        body=body,
        loop_vars=(i, output),
        shape_invariants=(i.shape, tf.TensorShape([None, None, None]))
      )

      return output

    outputs = {key: stack_and_concat(overall_outputs_for_key) for key, overall_outputs_for_key in
               overall_outputs_per_key.items()}

    print(f"overall_outputs_per_key={overall_outputs_per_key}")
    print(
      f"overall_outputs_per_key['f0_midi_dv_logits'].read(0)={overall_outputs_per_key['f0_midi_dv_logits'].read(0)}")
    print(
      f"overall_outputs_per_key['f0_midi_dv_logits'].stack()={overall_outputs_per_key['f0_midi_dv_logits'].stack()}")
    print(f"outputs={outputs}")
    print(f"end of MidiExpreToF0AutoregDecoder.autoregressive()")
    return outputs

  def autoregressive_original(self, cond, training=False, display_progressbar=False):
    """Autoregressive inference."""
    print(f"MidiExpreToF0AutoregDecoder.autoregressive()")
    cond_encode = self.encode_cond(cond)
    batch_size = tf.shape(cond_encode)[0]
    length = cond_encode.shape[1]
    prev_out = tf.tile([[0.0]], [batch_size, self.n_out])[:, tf.newaxis, :]  # go_frame
    prev_states = (None, None)
    overall_outputs = []

    print(f"MidiExpreToF0AutoregDecoder.autoregressive(): iterating, len={length}")

    for i in tqdm(range(length), position=0, leave=True, desc='Generating: ',
                  disable=not display_progressbar):
      print(f"MidiExpreToF0AutoregDecoder.autoregressive(): iterating, i={i}/{length}")
      curr_cond = cond_encode[:, i, :][:, tf.newaxis, :]
      prev_out = self.encode_out(prev_out, training=training)
      curr_out, curr_states = self._one_step(curr_cond, prev_out, prev_states, training=training)
      curr_out = self.sample_out(curr_out, training=training)
      overall_outputs.append(curr_out)
      prev_out, prev_states = curr_out['curr_out'], curr_states

    print(f"overall_outputs={overall_outputs}")

    outputs = {}
    for k in curr_out.keys():
      outputs[k] = tf.concat([x[k] for x in overall_outputs], 1)

    print(f"outputs={outputs}")
    print(f"end of MidiExpreToF0AutoregDecoder.autoregressive()")
    return outputs

  def right_shift_encode_out(self, out, training=False):
    """Right shift the ground truth target by one timestep and encode."""
    out_dynamic_shape = tf.shape(out)
    go_frame = tf.tile([[0.0]], [out_dynamic_shape[0], out_dynamic_shape[-1]])[:, tf.newaxis, :]
    out = tf.concat([go_frame, out[:, :-1, :]], 1)
    out = self.encode_out(out, training=training)
    # input dropout
    if self.input_dropout:
      out_dynamic_shape = tf.shape(out)
      input_dropout_mask = tf.random.uniform([out_dynamic_shape[0], out_dynamic_shape[1], 1]) > self.input_dropout_p
      input_dropout_mask = tf.cast(input_dropout_mask, tf.float32)
      out = out * input_dropout_mask
    return out

  def teacher_force(self, cond, out, training=True):
    """Run teacher force."""
    print(f"MidiExpreToF0AutoregDecoder.teacher_force()")
    out_shifted = self.right_shift_encode_out(out, training=training)
    cond = self.encode_cond(cond, training=training)
    z_in = tf.concat([cond, out_shifted], -1)
    z_in = self.encode_z(z_in, training=training)
    z_out, *states = self.rnn1(z_in, training=training)
    z_out, *states = self.rnn2(z_out, training=training)
    output = self.decode_out(z_out, training=training)
    output = self.split_teacher_force_output(output)
    print(f"end of MidiExpreToF0AutoregDecoder.teacher_force()")
    return output

  def encode_z(self, z, training=False):
    return z

  def encode_out(self, out, training=False):
    return out

  def encode_cond(self, cond, training=False):
    return cond

  def decode_out(self, z_out, training=False):
    return z_out

  def sample_out(self, out, training=False):
    return out

  def preprocess(self, cond, out):
    return cond, out

  def postprocess(self, outputs, cond, training=False):
    return outputs

  def split_teacher_force_output(self, output):
    return output

  def call(self, cond, out=None, training=False):
    """Forward call."""

    print("TwoLayerCondAutoregRNN.call()")

    if training:
      cond, out = self.preprocess(cond, out)
      outputs = self.teacher_force(cond, out, training=training)
    else:
      cond, out = self.preprocess(cond, out)
      outputs = self.autoregressive(cond, training=training)

    outputs = self.postprocess(outputs, cond, training=training)

    return outputs
