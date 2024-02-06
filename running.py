from bark import SAMPLE_RATE, generate_audio, preload_models
import soundfile as sf

# download and load all models
preload_models()

# generate audio from text
texts = """
It’s Sarah from ZOOM Realty Group. How is your day going?
I'm sorry to hear that. I'm calling because I noticed that you were looking for homes for sale recently. Can I talk to you about that?
Great! So, I am on a recorded line and the reason I’m calling you is to find a time for you to chat with ZOOM Realty Group. What area are you looking for a home in?
Got it. I apologize for the confusion. So, you're looking to sell your home. Do you have a timeframe for when you'll want to sell?
Awesome. So your home is already on the market. Got it. And do you have a timeframe for when you'll want to sell?
Understood. So you're looking to sell your home as soon as possible. We can definitely help with that. Let's go ahead and schedule a call with someone on our team. Does today, tomorrow, or the day after tomorrow work best for your call?
Okay, let's schedule a call to discuss your house. What time works best for you on that day?
"""
prompt_texts = texts.split('\n')
history_prompt = ['hey_james_494'] * len(prompt_texts)

audio_array = generate_audio(
    prompt_texts,
    history_prompt=history_prompt
)
print(audio_array.shape)
import numpy as np
# save audio to disk
for i in range(len(prompt_texts)):
    sf.write(f"bark_generation_{i}.wav", audio_array[i], SAMPLE_RATE)
