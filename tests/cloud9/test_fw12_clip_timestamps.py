from speaches.executors import whisper
merged=[{"start":16000,"end":48000,"segments":[(16000,24000),(32000,48000)]}]
assert whisper._clip_timestamps_seconds(merged)==[{"start":1.0,"end":3.0}]
print("FW12_CLIP_TIMESTAMPS=PASS")
