# WaveNet vocoder

A Pytorch implementation of the WaveNet vocoder, which can generate raw speech samples conditioned on mel spectrograms.
This task refers to a speech synthesis problem, when we need to reconstruct an audio signal from a mel spectrogram.

## Usage

You can download my pretrained model or train your own. Settings for calculating mel spectrograms can be found here:

```python
from config import MelSpectrogramConfig
from src.preprocessing import MelSpectrogram

featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
mel_spectrogram = featurizer(audio_wav)
```


Then, prediction:

```python
predicted_audio = model.inference(mel_spectrogram)
```
