import time
import torch
import soundfile as sf
from transformers import BarkModel
from transformers import AutoProcessor

device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("suno/bark-small")

model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)

# convert to bettertransformer
model = model.to_bettertransformer()


text_prompt = [
    "Let's try generating speech, with Bark, a text-to-speech model",
    "Wow, batching is so great!",
    "I love Hugging Face, it's so cool.",
    "I love Hugging Face, it's so cool.",
    "I love Hugging Face, it's so cool.",
    "I love Hugging Face, it's so cool.",
    "I love Hugging Face, it's so cool.",
    "I love Hugging Face, it's so cool.",
]

inputs = processor(text_prompt).to(device)
s = time.time()
with torch.inference_mode():
    # samples are generated all at once
    speech_output = model.generate(**inputs, do_sample=True, fine_temperature=0.4, coarse_temperature=0.8, return_output_lengths=True)
generation_time = time.time() - s

speech_output = speech_output.cpu().float().numpy()
for i in range(len(text_prompt)):
    sf.write(f'output/{i}.wav', speech_output[i, :], 24000)
total_duration = speech_output.shape[1] / 24000

print(generation_time)
print(total_duration)
print(generation_time / total_duration)
