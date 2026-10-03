#!/usr/bin/env python3
import argparse, json, pathlib, re, subprocess, threading, time, unicodedata, urllib.parse

DEFAULT_MODELS = [
    ("fr-fast", "handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf"),
    ("fr-quality", "handy-computer/Qwen3-ASR-1.7B-gguf"),
    ("stt-whisper", "deepdml/faster-whisper-large-v3-turbo-ct2"),
    ("fr-small", "Systran/faster-whisper-small"),
]

def normalize(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s.casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("’", " ").replace("'", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.split()

def wer(ref: str, hyp: str):
    r, h = normalize(ref), normalize(hyp)
    dp = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev = dp[0]
        dp[0] = i
        for j, hw in enumerate(h, 1):
            old = dp[j]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (rw != hw))
            prev = old
    edits = dp[-1]
    return edits / max(1, len(r)), edits, len(r)

def read_int(path):
    try:
        return int(pathlib.Path(path).read_text().strip())
    except Exception:
        return None

def sample_memory(stop, peaks):
    paths = {
        "gtt_used": "/sys/class/drm/card0/device/mem_info_gtt_used",
        "vram_used": "/sys/class/drm/card0/device/mem_info_vram_used",
    }
    while not stop.is_set():
        for k,p in paths.items():
            v=read_int(p)
            if v is not None: peaks[k]=max(peaks.get(k,0),v)
        stop.wait(0.05)

def transcribe(base, audio, model):
    peaks={}
    stop=threading.Event()
    t=threading.Thread(target=sample_memory,args=(stop,peaks),daemon=True)
    t.start()
    started=time.monotonic()
    cp=subprocess.run([
        "curl","-sS","--fail-with-body","--max-time","180",
        f"{base}/v1/audio/transcriptions",
        "-F",f"file=@{audio};type=audio/wav",
        "-F",f"model={model}",
        "-F","language=fr",
        "-F","response_format=json",
    ],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    wall=time.monotonic()-started
    stop.set(); t.join(timeout=1)
    if cp.returncode:
        raise RuntimeError(f"curl rc={cp.returncode}: {cp.stderr[-2000:]} body={cp.stdout[-2000:]}")
    try:
        obj=json.loads(cp.stdout)
        text=obj.get("text") or obj.get("transcript") or cp.stdout
    except json.JSONDecodeError:
        text=cp.stdout.strip()
    return text.strip(), wall, peaks

def unload(base, model):
    mid=urllib.parse.quote(model, safe="/")
    cp=subprocess.run(["curl","-sS","--max-time","10","-X","DELETE",f"{base}/api/ps/{mid}"],
                      text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return {"rc":cp.returncode,"response":cp.stdout.strip()[-1000:]}

def wav_duration(path):
    import wave
    with wave.open(path,"rb") as w:
        return w.getnframes()/w.getframerate()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-url",default="http://127.0.0.1:8005")
    ap.add_argument("--audio",required=True)
    ap.add_argument("--reference",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--passes",type=int,default=2)
    ap.add_argument("--model",action="append",help="label=model_id; repeatable")
    a=ap.parse_args()
    models=DEFAULT_MODELS
    if a.model:
        models=[]
        for x in a.model:
            label,mid=x.split("=",1); models.append((label,mid))
    ref=pathlib.Path(a.reference).read_text().strip()
    dur=wav_duration(a.audio)
    out={"created_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"fixture":a.audio,
         "reference":ref,"duration_s":dur,"normalization":"casefold, NFKD/strip diacritics, punctuation/apostrophe/hyphen insensitive",
         "models":[]}
    for label,mid in models:
        item={"label":label,"model":mid,"runs":[]}
        for n in range(1,a.passes+1):
            text,wall,peaks=transcribe(a.base_url,a.audio,mid)
            w,e,nref=wer(ref,text)
            run={"pass":n,"mode":"cold" if n==1 else "warm","wall_s":wall,
                 "rtf":wall/dur,"wer":w,"word_edits":e,"reference_words":nref,
                 "text":text,
                 "peak_gtt_bytes":peaks.get("gtt_used"),"peak_vram_bytes":peaks.get("vram_used")}
            item["runs"].append(run)
            print(json.dumps({"label":label,**run},ensure_ascii=False),flush=True)
        item["unload"]=unload(a.base_url,mid)
        time.sleep(1)
        out["models"].append(item)
    pathlib.Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print("RESULT",a.output)

if __name__=="__main__":
    main()
