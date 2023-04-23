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

"""Model class for Synthesis Generator + DDSP Inference module."""

import tensorflow as tf
import ddsp
import ddsp.training

from midi_ddsp.utils.inference_utils import get_process_group
from .interpretable_conditioning import normalize_synth_params, extract_harm_controls


class SynthCoder(tf.keras.Model):
  """The DDSP Inference module."""

  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.model_type = 'SynthCoder'

  def encode(self, inputs, training=False):
    z = self.encoder(inputs, training=training)
    return z

  def decode(self, z, inputs):
    synth_params = self.decoder([z, inputs])
    return synth_params

  def call(self, inputs, training=None):
    print("SynthCoder.__call__()")
    z = self.encode(inputs, training=training)
    synth_params = self.decode(z, inputs)
    return synth_params

  def _build(self, inputs):
    inputs, kwargs = inputs
    self(inputs, **kwargs)


class MIDIExpressionAE(tf.keras.Model):
  """
  The DDSP Inference + Synthesis Generator module.
  The naming of the module is not exactly the same as in the MIDI-DDSP paper,
  but one can find correspondence in following crude diagram:

                                       Note Sequence
                                             │
          Note Expression Controls           │       │
            (conditioning_dict)    ──────────┤       │
                    ▲                        │       │
      Feature       │                        ├────┐  │
     Extraction     │                        │    │  │
  (gen_cond_dict)   │                        ▼    │  │ Synthesis Generator
           Synthesis Parameters             f0    │  │   (midi_decoder)
            (f0,amps,hd,noise)               │    │  │
                    ▲                        ├────┘  │
      DDSP          │                        │       │
   Inference        │                        │       │
  (SynthCoder)      │                        │       │
                    │                        ▼
              Audio Feature         Synthesis Parameters
               (f0,ld,mel)           (f0,amps,hd,noise)
                    ▲                        │
      CREPE         │                        │
   A-Weighting      │                        │ DDSP
     Mel-STFT       │                        │
                    │                        ▼
                  Audio                    Audio
                  Input               Reconstruction
  """

  def __init__(self, synth_coder, midi_decoder, processor_group, n_frames=1000, frame_size=64,
               sample_rate=16000, reverb_module=None,
               use_f0_ld=False, run_synth_coder_only=False, vst_inference_mode=False):
    super().__init__()
    self.synth_coder = synth_coder
    self.midi_decoder = midi_decoder
    self.reverb_module = reverb_module
    self.model_type = 'MIDIExpressionAE'


    self.processor_group = processor_group
    self.n_frames = n_frames
    self.frame_size = frame_size
    self.sample_rate = sample_rate

    self.use_f0_ld = use_f0_ld

    self.run_synth_coder_only = run_synth_coder_only
    # Only works when running whole model
    self.run_without_synths = vst_inference_mode
    # Structures output in a format that is suitable for vst
    self.run_inside_vst = vst_inference_mode

  def train_synth_coder_only(self):
    self.midi_decoder.trainable = False
    self.synth_coder.trainable = True

  def freeze_synth_coder(self):
    self.midi_decoder.trainable = True
    self.synth_coder.trainable = False
    if self.reverb_module is not None:
      self.reverb_module.trainable = False

  def run_synth_coder(self, features, run_without_synths, training=True):
    synth_params = self.synth_coder(features, training=training)

    control_params = None
    synth_audio = None

    if run_without_synths is False:
      control_params = self.processor_group.get_controls(synth_params, verbose=False)
      synth_audio = self.processor_group.get_signal(control_params)

      if self.reverb_module is not None:
        synth_audio = self.reverb_module(synth_audio, reverb_number=features['instrument_id'], training=training)

    return synth_params, control_params, synth_audio

  @staticmethod
  def get_gt_midi(features):
    """Prepare the ground truth MIDI."""
    f0_midi = ddsp.core.hz_to_midi(features['f0_hz'])
    q_pitch = tf.cast(features['midi'][..., tf.newaxis], tf.float32)

    q_vel = q_pitch * 0.0  # (Jesse) Don't use velocities for now.

    f0_loss_weights = tf.cast(tf.abs(f0_midi - q_pitch) < 2.0, tf.float32)

    return q_pitch, q_vel, f0_loss_weights, features['onsets'], features['offsets']

  def gen_cond_dict_from_feature(self, features, training=False):
    synth_params, control_params, synth_audio = self.run_synth_coder(features, run_without_synths=self.run_without_synths, training=training)

    # synth_params_normalized: scaled and normalized synth params
    synth_params_normalized = normalize_synth_params(synth_params)

    midi_features = self.get_gt_midi(features)

    conditioning_dict = self.midi_decoder.gen_interpretable_conditioning_dict(synth_params_normalized, midi_features)

    return synth_params_normalized, midi_features, conditioning_dict

  def gen_audio_from_cond_dict(self, conditioning_dict, midi_features,
                               instrument_id=None,
                               use_angular_cumsum=True,
                               display_progressbar=False):
    z_midi_decoder, params_pred = self.midi_decoder.generate_synth_params_from_interpretable_conditioning(
      conditioning_dict, midi_features,
      instrument_id=instrument_id,
      display_progressbar=display_progressbar)

    self.processor_group = get_process_group(z_midi_decoder.shape[1],
                                             self.frame_size, self.sample_rate,
                                             use_angular_cumsum=
                                             use_angular_cumsum)

    if self.use_f0_ld:
      midi_synth_params, midi_control_params, midi_audio = self.run_synth_coder(
        {'f0_hz': params_pred['f0_hz'],
         'loudness_db': params_pred['ld'],
         'instrument_id': instrument_id},
        run_without_synths=self.run_without_synths,
        training=False)
      f0_pred, amps_pred, hd_pred, noise_pred = normalize_synth_params(midi_synth_params)

    else:
      midi_synth_params = self.processor_group.get_controls(
        {'amplitudes': params_pred['amplitudes'],
         'harmonic_distribution': params_pred[
           'harmonic_distribution'],
         'noise_magnitudes': params_pred['noise_magnitudes'],
         'f0_hz': params_pred['f0_hz'], },
        verbose=False)

      midi_audio = self.processor_group.get_signal(midi_synth_params)
      if self.reverb_module is not None:
        midi_audio = self.reverb_module(midi_audio, reverb_number=instrument_id, training=False)
      f0_pred, amps_pred, hd_pred, noise_pred = normalize_synth_params(midi_synth_params)

    midi_control_params = (f0_pred, amps_pred, hd_pred, noise_pred)
    return midi_audio, midi_control_params, midi_synth_params

  def call(self, features, training=False, run_synth_coder_only=None):
    print("MIDIExpressionAE.__call__()")

    """Run the network to get a prediction and optionally get losses."""

    if run_synth_coder_only is not None:
      self.run_synth_coder_only = run_synth_coder_only

    synth_params, control_params, synth_audio = self.run_synth_coder(features, run_without_synths=self.run_without_synths, training=training)
    if self.run_synth_coder_only:  # only run synth coder branch
      if self.run_without_synths:
        outputs = {
          'synth_params': synth_params,
        }
      else:
        outputs = {
          'synth_params': synth_params,
          'control_params': control_params,
          'synth_audio': synth_audio,
        }
      return outputs

    # synth_params_normalized: scaled and normalized synth params
    synth_params_normalized = normalize_synth_params(synth_params, stop_gradient=True)
    #synth_params_normalized = extract_harm_controls(control_params, stop_gradient=True)

    # rearrange midi features in readable format
    midi_features = self.get_gt_midi(features)

    # --- MIDI Decoding
    conditioning_dict, params_pred = self.midi_decoder(features,
                                                       synth_params_normalized,
                                                       midi_features,
                                                       training=training,
                                                       synth_params=synth_params)

    if self.run_without_synths:
      midi_synth_params = {'harmonic_distribution': params_pred['harmonic_distribution'],
        'f0_hz': params_pred['f0_hz'],
        # Omitting 'state' here. NB! MIDI-DDSP expects it!
        'amplitudes': params_pred['amplitudes'],
        'noise_magnitudes': params_pred['noise_magnitudes']
       }

      outputs = {
        'synth_params': synth_params,
        'midi_synth_params': midi_synth_params,

        'midi_features': midi_features,
        'conditioning_dict': conditioning_dict,
      }
    else:

      if self.use_f0_ld:
        midi_synth_params, midi_control_params, midi_audio = self.run_synth_coder(
          {'f0_hz': params_pred['f0_hz'],
           'loudness_db': params_pred['ld'],
           'instrument_id': features['instrument_id']},
          run_without_synths=self.run_without_synths,
          training=training)

      else:
        midi_synth_params = {
          'amplitudes': params_pred['amplitudes'],
          'harmonic_distribution': params_pred['harmonic_distribution'],
          'noise_magnitudes': params_pred['noise_magnitudes'],
          'f0_hz': params_pred['f0_hz']}

        midi_control_params = self.processor_group.get_controls(midi_synth_params, verbose=False)
        # --- MIDI Audio Losses
        midi_audio = self.processor_group.get_signal(midi_control_params)

        if self.reverb_module is not None:
          midi_audio = self.reverb_module(midi_audio,
                                          reverb_number=features['instrument_id'],
                                          training=training)

      # unpack things
      f0_pred, amps_pred, hd_pred, noise_pred = normalize_synth_params(midi_synth_params)
      q_pitch, q_vel, f0_loss_weights, onsets, offsets = midi_features
      f0_synthcoder, amps_synthcoder, hd_synthcoder, noise_synthcoder = synth_params_normalized

      # --- Finalize and return
      outputs = {
        'synth_params': synth_params,
        'control_params': control_params,
        'synth_audio': synth_audio,

        'midi_synth_params': midi_synth_params,
        'midi_audio': midi_audio,

        'midi_features': midi_features,
        'conditioning_dict': conditioning_dict,

        'amps': amps_synthcoder,
        'hd': hd_synthcoder,
        'noise': noise_synthcoder,
        'amps_pred': amps_pred,
        'hd_pred': hd_pred,
        'noise_pred': noise_pred,
        'f0_loss_weights': f0_loss_weights,
        'q_pitch': q_pitch,
        'q_vel': q_vel,
        'params_pred': params_pred,
      }

    return outputs

  def _build(self, inputs):
    inputs, kwargs = inputs
    self(inputs, **kwargs)


class MIDIExpressionAE_VST_IO_Wrapper(tf.keras.Model):

  """
  Wrapper that generates all the additional inputs from the audio
  and makes the MIDIExpressionAE compatible with VST input/output interface
  """

  def __init__(self, ae_model):
    tf.keras.Model.__init__(self)

    self.ae = ae_model

  #def __getattr__(self, item):
  #  print(f"__getattr__ trying to get item={item}")
  #  return getattr(self.ae_model, item)

  def _unpack_synth_params(self, params):
    return {
      'amplitudes': params['amplitudes'],
      'harmonic_distribution': params['harmonic_distribution'],
      'noise_magnitudes': params['noise_magnitudes'],
      'f0_hz': params['f0_hz']
    }

  def call(self, features, training=False, run_synth_coder_only=None):
    print("MIDIExpressionAE_VST_IO_Wrapper.__call__()")

    results = self.ae(features, training=training, run_synth_coder_only=run_synth_coder_only)

    if self.ae.run_synth_coder_only:
      return self._unpack_synth_params(results['synth_params'])
    else:
      return self._unpack_synth_params(results['midi_synth_params'])


