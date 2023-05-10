from math import ceil

import tensorflow as tf
from ddsp import core


class MIDIExpressionAE_VST_IO_Wrapper(tf.keras.Model):

  """
  Wrapper that generates all the additional inputs from the audio
  and makes the MIDIExpressionAE compatible with VST input/output interface
  """

  def __init__(self, ae_model, vst_buffer_size, vst_frame_size, vst_scale_outputs=False):
    tf.keras.Model.__init__(self)

    self.ae = ae_model

    self.vst_buffer_size = vst_buffer_size
    self.vst_frame_size = vst_frame_size
    self.vst_num_frames_in_buffer = ceil(self.vst_buffer_size / self.vst_frame_size)

    self.vst_scale_outputs = vst_scale_outputs

    # Additional stuff to make output conversions like in original DDSPv1 VST model. It can be redundant
    self.scale_fn = core.exp_sigmoid
    self.sample_rate = 16000
    self.initial_bias = -5.0
  #def __getattr__(self, item): # it breaks the model
  #  logging.debug(f"__getattr__ trying to get item={item}")
  #  return getattr(self.ae_model, item)

  @property
  def _signatures(self):
    return {'call': self.call.get_concrete_function(
        audio=        tf.TensorSpec(shape=[1024], dtype=tf.float32),
        loudness_db=  tf.TensorSpec(shape=[1], dtype=tf.float32),
        f0_hz=        tf.TensorSpec(shape=[1], dtype=tf.float32),
        midi=         tf.TensorSpec(shape=[1], dtype=tf.float32),
        onsets=       tf.TensorSpec(shape=[1], dtype=tf.float32),
        offsets=      tf.TensorSpec(shape=[1], dtype=tf.float32),
        instrument_id=tf.TensorSpec(shape=[1], dtype=tf.float32),
        state=        tf.TensorSpec(shape=[self.state_size], dtype=tf.float32),
    )}

  def _unpack_synth_params(self, params):
    return {
      'amplitudes': params['amplitudes'],
      'harmonic_distribution': params['harmonic_distribution'],
      'noise_magnitudes': params['noise_magnitudes'],
      'f0_hz': params['f0_hz']
    }

  def _get_controls(self,
                   amplitudes,
                   harmonic_distribution,
                   f0_hz):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      amplitudes: 3-D Tensor of synthesizer controls, of shape
        [batch, time, 1].
      harmonic_distribution: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_harmonics].
      f0_hz: Fundamental frequencies in hertz. Shape [batch, time, 1].

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the amplitudes.
    if self.scale_fn is not None:
      amplitudes = self.scale_fn(amplitudes)
      harmonic_distribution = self.scale_fn(harmonic_distribution)

    harmonic_distribution = core.normalize_harmonics(harmonic_distribution, f0_hz, self.sample_rate)
        #self.sample_rate if self.normalize_below_nyquist else None)

    return {'amplitudes': amplitudes,
            'harmonic_distribution': harmonic_distribution,
            'f0_hz': f0_hz}

  def _get_noise_controls(self, magnitudes):
    """Convert network outputs into a dictionary of synthesizer controls.

    Args:
      magnitudes: 3-D Tensor of synthesizer parameters, of shape [batch, time,
        n_filter_banks].

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the magnitudes.
    if self.scale_fn is not None:
      magnitudes = self.scale_fn(magnitudes + self.initial_bias)

    return {'magnitudes': magnitudes}


  def call(self, audio, loudness_db, f0_hz, midi, onsets, offsets, instrument_id, state,
           training=False, run_synth_coder_only=None):
    logging.debug("MIDIExpressionAE_VST_IO_Wrapper.__call__()")

    audio = tf.reshape(audio, [1, self.vst_buffer_size])
    loudness_db = tf.reshape(loudness_db, [1, self.vst_num_frames_in_buffer, 1])
    f0_hz = tf.reshape(f0_hz, [1, self.vst_num_frames_in_buffer, 1])  # TODO: or is it f0_scaled?
    midi = tf.reshape(midi, [1, self.vst_num_frames_in_buffer])

    onsets = tf.reshape(onsets, [1, self.vst_num_frames_in_buffer])
    offsets = tf.reshape(offsets, [1, self.vst_num_frames_in_buffer])
    #instrument_id = instrument_id

    #state = tf.reshape(state, [1, self.state_size])

    reshaped_inputs = {
      'audio': audio,
      'loudness_db': loudness_db,
      'f0_hz': f0_hz,
      'midi': midi,
      'onsets': onsets,
      'offsets': offsets,
      'instrument_id': instrument_id,
    }

    logging.debug(reshaped_inputs)

    outputs = self.ae(reshaped_inputs, training=training, run_synth_coder_only=run_synth_coder_only)

    synth_params = None
    if self.ae.run_synth_coder_only:
      synth_params = self._unpack_synth_params(outputs['synth_params'])
    else:
      synth_params = self._unpack_synth_params(outputs['midi_synth_params'])

    if self.vst_scale_outputs:
    # Apply the nonlinearities.
      harm_controls = self._get_controls(
                              amplitudes=synth_params['amplitudes'],
                              harmonic_distribution=synth_params['harmonic_distribution'],
                              f0_hz=f0_hz)

      noise_controls = self._get_noise_controls(magnitudes=synth_params['noise_magnitudes'])
    else:
      harm_controls = synth_params['harmonic_distribution']
      noise_controls = synth_params['noise_magnitudes']

    # Return 1-D tensors.
    amps = tf.reshape(synth_params['amplitudes'][0][:self.vst_num_frames_in_buffer], [self.vst_num_frames_in_buffer])

    hd = noise_controls[0][:self.vst_num_frames_in_buffer]
    if self.vst_num_frames_in_buffer == 1:
        hd = tf.reshape(hd, [hd.shape[1]])

    noise = harm_controls[0][:self.vst_num_frames_in_buffer]
    if self.vst_num_frames_in_buffer == 1:
        noise = tf.reshape(noise, [noise.shape[1]])

    state = state # do nothing for now, we don't use stateless RNNs yet

    return {
      'amplitudes': amps,
      'harmonic_distribution': hd,
      'noise_magnitudes': noise,
      'state': state
    }
