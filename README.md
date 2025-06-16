# CodecEncoderDecoder

***
***
```
git clone https://github.com/CodeVault-girish/NeuralCodecDecoder.git
cd NeuralCodecDecoder
```
## Recomend create seprate env for each codec model
```
from audio_codec.registry import CODEC_REGISTRY
from audio_codec.cli import decoder_list, decode_folder
```

# list available

```
decoder_list()
```

---
# Supported Models codec
## Available decoders

The following codec decoders are available in this repository:

1. **snac_24khz**  
2. **snac_32khz**  
3. **snac_44khz**  
4. **dac_16khz**  
5. **dac_24khz**  
6. **dac_44khz**  
7. **encodec_24khz**  
8. **encodec_48khz**  
9. **soundstream_16khz**  
10. **speechtokenizer**
--. **Funcodec**
--. **AudioDec**
```
decode_folder('2', 'raw_wavs', 'decoded', 'cpu')
decode_folder('10', '/home/girish/Girish/Reseach/Health-care/Audio_Data/Audio_Data/HC', 'output/', 'cuda')
```
---
---
# For SpeechTokenizer 
## use this link to get it's model 

[fnlp/SpeechTokenizer model](https://huggingface.co/fnlp/SpeechTokenizer/tree/main/speechtokenizer_hubert_avg)

Add this model file as follows:

```
NeuralCodecDecoder/
  audio_codec/
    codecs/
  config/
    config.json
  checkpoints/
    SpeechTokenizer.pt
```

Place the downloaded `SpeechTokenizer.pt` file into the `checkpoints/` directory as shown above.

```
pip install --upgrade pip setuptools wheel
pip install --only-binary=:all: tokenizers

pip install git+https://github.com/ga642381/AudioCodec-Hub.git soundfile
```
additional
```
pip install --no-deps --force-reinstall git+https://github.com/ga642381/AudioCodec-Hub.git soundfile
```
---
---
## For Funcodec
```
pyhton3 -m venv funcodec
source funcodec/bin/activate
```
```
git clone https://github.com/alibaba-damo-academy/FunCodec.git
cd FunCodec
pip install -e .
```
```
pip install torch torchaudio numpy soundfile
```
```
cd egs/LibriTTS/codec
mkdir -p exp
model_name="audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch"
# Download the model
git lfs install
git clone https://huggingface.co/alibaba-damo/${model_name}
```
## folder path
```
find ../../../../../Girish/Reseach/Health-care/Audio_Data/Audio_Data/HC/  -name "*.wav" | awk -F/ '{printf "%s %s\n", $(NF-1) "_" $NF, $0}' > input.scp
```
# Encoding
```
model_name=audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch
bash encoding_decoding.sh \
  --stage 1 \
  --batch_size 1 \
  --num_workers 1 \
  --gpu_devices "0" \
  --model_dir exp/${model_name} \
  --bit_width 16000 \
  --file_sampling_rate 16000 \
  --wav_scp input.scp \
  --out_dir outputs/codecs
```
# Decoding
```
model_name=audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch
bash encoding_decoding.sh \
  --stage 2 \
  --batch_size 1 \
  --num_workers 1 \
  --gpu_devices "0" \
  --model_dir exp/${model_name} \
  --bit_width 16000 \
  --file_sampling_rate 16000 \
  --wav_scp outputs/codecs/codecs.txt \
  --out_dir outputs/recon_wavs
```

---
---
# AudioDec

## take AudioDec.py file from this repo and paste in 
```
git clone https://github.com/facebookresearch/AudioDec.git
cd AudioDec
pip install -r requirements.txt
```
## Download this [exp](https://github.com/facebookresearch/AudioDec/releases/download/pretrain_models_v02/exp.zip) and save in this

<!-- python demoFile.py --model vctk_v1  -i ../codec/test/ -o output/
python demoFile.py --model libritts_v1 -i ../codec/test/ -o output/
python demoFile.py --model libritts_v1 -i ../codec/test/A002_02_BBP_NORMAL.wav -o output.wav

 -->
## Usage

### 1. Single-file mode

- **With GPU** (e.g., CUDA device 0)  
  ```
  python demoFile.py \\
    --model libritts_v1 \\
    -i path/to/input.wav \\
    -o path/to/output.wav
  ```

- **With CPU only**  
  ```
  python demoFile.py \\
    --cuda -1 \\
    --model vctk_v1 \\
    -i path/to/input.wav \\
    -o path/to/output.wav
  ```

### 2. Folder mode

- **With GPU**  
  ```
  python demoFile.py \\
    --model vctk_v1 \\
    -i path/to/input_folder \\
    -o path/to/output_folder
  ```

- **With CPU only**  
  ```
  python demoFile.py \\
    --cuda -1 \\
    --model libritts_v1 \\
    -i path/to/input_folder \\
    -o path/to/output_folder
  ```
