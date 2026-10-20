"""
ShortForge AI — весь бэкенд в одном файле.
Запуск: pip install fastapi uvicorn fastapi-cors && python app.py
Открыть: http://127.0.0.1:8765
"""
import subprocess, uuid, os, random, asyncio
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

OUT = Path.home() / "Movies" / "ShortForge"
OUT.mkdir(parents=True, exist_ok=True)
JOBS = {}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ============ МОДЕЛИ ============

class CutReq(BaseModel):
    files: list[str]
    length: int = 30
    count: int = 5

class UniqueReq(BaseModel):
    files: list[str]
    copies: int = 5
    intensity: int = 50

class TransReq(BaseModel):
    file: str
    src: str = "en"
    dst: str = "ru"

class TTSReq(BaseModel):
    voice: str = "alexey"
    text: str = "Пример озвучки ShortForge AI."

# ============ ВСПОМОГАТЕЛЬНЫЕ ============

def ffprobe_dur(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=10
        )
        return float(r.stdout.strip() or 0)
    except:
        return 0

def cut_one(src, start, dur, out, fmt="9:16"):
    vf = {"9:16": "crop=ih*9/16:ih,scale=1080:1920",
          "1:1": "crop=ih:ih,scale=1080:1080",
          "16:9": "scale=1920:1080"}.get(fmt, "crop=ih*9/16:ih,scale=1080:1920")
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", src, "-t", str(dur),
           "-vf", vf, "-c:v", "libx264", "-crf", "22", "-preset", "fast",
           "-c:a", "aac", "-b:a", "128k", out]
    subprocess.run(cmd, check=True, capture_output=True, timeout=600)

def run_cut(req: CutReq, job_id: str):
    JOBS[job_id] = {"status": "running", "progress": 0, "outputs": [], "errors": []}
    total = len(req.files) * req.count
    done = 0
    for f in req.files:
        if not os.path.exists(f):
            JOBS[job_id]["errors"].append(f"Файл не найден: {f}")
            continue
        dur = ffprobe_dur(f)
        if dur < req.length:
            JOBS[job_id]["errors"].append(f"Слишком короткое: {f}")
            continue
        step = max(1, int(dur / req.count))
        for i in range(req.count):
            start = i * step
            name = f"{Path(f).stem}_short_{i+1}.mp4"
            out = str(OUT / name)
            try:
                cut_one(f, start, req.length, out)
                JOBS[job_id]["outputs"].append(name)
            except Exception as e:
                JOBS[job_id]["errors"].append(str(e))
            done += 1
            JOBS[job_id]["progress"] = int(done / total * 100)
    JOBS[job_id]["status"] = "done"

def run_unique(req: UniqueReq, job_id: str):
    JOBS[job_id] = {"status": "running", "progress": 0, "outputs": [], "errors": []}
    total = len(req.files) * req.copies
    done = 0
    k = req.intensity / 100.0
    for f in req.files:
        if not os.path.exists(f):
            JOBS[job_id]["errors"].append(f"Файл не найден: {f}")
            continue
        for i in range(req.copies):
            name = f"{Path(f).stem}_uniq_{i+1}.mp4"
            out = str(OUT / name)
            try:
                speed = 1 + random.uniform(-0.05, 0.05) * k
                hue = random.uniform(-5, 5) * k
                noise = int(1 + random.random() * 4 * k)
                filters = [f"setpts=PTS/{speed}"]
                if random.random() < 0.5: filters.append("hflip")
                filters.append("crop=iw*0.98:ih*0.98,scale=iw:ih")
                filters.append(f"eq=hue={hue}:saturation=1.0:contrast=1.0")
                if noise > 0: filters.append(f"noise=alls={noise}:allf=t")
                cmd = ["ffmpeg", "-y", "-i", f, "-vf", ",".join(filters),
                       "-af", f"atempo={speed}", "-map_metadata", "-1",
                       "-c:v", "libx264", "-crf", "22", "-c:a", "aac", "-b:a", "128k", out]
                subprocess.run(cmd, check=True, capture_output=True, timeout=600)
                JOBS[job_id]["outputs"].append(name)
            except Exception as e:
                JOBS[job_id]["errors"].append(str(e))
            done += 1
            JOBS[job_id]["progress"] = int(done / total * 100)
    JOBS[job_id]["status"] = "done"

def run_translate(req: TransReq, job_id: str):
    JOBS[job_id] = {"status": "running", "progress": 0, "outputs": [], "errors": [], "segments": []}
    try:
        from faster_whisper import WhisperModel
        JOBS[job_id]["progress"] = 20
        model = WhisperModel("small", device="auto", compute_type="int8")
        segments, _ = model.transcribe(req.file, language=req.src, vad_filter=True)
        seg_list = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
        JOBS[job_id]["progress"] = 70

        # Простой перевод через Argos (офлайн)
        translated = []
        try:
            from argostranslate import translate as argo_trans
            for s in seg_list:
                tr = argo_trans.translate(s["text"], req.src, req.dst) or s["text"]
                translated.append({"src": s["text"], "dst": tr})
        except:
            # Fallback — просто помечаем
            translated = [{"src": s["text"], "dst": f"[{req.dst}] {s['text']}"} for s in seg_list]

        JOBS[job_id]["segments"] = translated
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["status"] = "done"
    except Exception as e:
        JOBS[job_id]["errors"].append(str(e))
        JOBS[job_id]["status"] = "done"

def run_tts(req: TTSReq, job_id: str):
    JOBS[job_id] = {"status": "running", "progress": 0, "outputs": [], "errors": []}
    try:
        import edge_tts
        voices = {"alexey": "ru-RU-DmitryNeural", "maria": "ru-RU-SvetlanaNeural"}
        voice = voices.get(req.voice, voices["alexey"])
        out = str(OUT / f"tts_{job_id[:6]}.mp3")
        asyncio.run(edge_tts.Communicate(req.text, voice).save(out))
        JOBS[job_id]["outputs"].append(Path(out).name)
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["status"] = "done"
    except Exception as e:
        JOBS[job_id]["errors"].append(str(e))
        JOBS[job_id]["status"] = "done"

# ============ API ============

@app.get("/api/health")
def health():
    ff = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    return {"ok": True, "ffmpeg": ff, "out_dir": str(OUT)}

@app.get("/api/out-files")
def out_files():
    files = []
    for f in sorted(OUT.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
        files.append({"name": f.name, "size_mb": round(f.stat().st_size / 1e6, 1)})
    return {"files": files}

@app.get("/api/download/{name}")
def download(name: str):
    p = OUT / name
    if p.exists():
        return FileResponse(p, filename=name)
    return {"error": "not found"}

@app.post("/api/cut")
def cut(req: CutReq, bg: BackgroundTasks):
    jid = uuid.uuid4().hex
    bg.add_task(run_cut, req, jid)
    return {"job_id": jid}

@app.post("/api/unique")
def unique(req: UniqueReq, bg: BackgroundTasks):
    jid = uuid.uuid4().hex
    bg.add_task(run_unique, req, jid)
    return {"job_id": jid}

@app.post("/api/translate")
def translate(req: TransReq, bg: BackgroundTasks):
    jid = uuid.uuid4().hex
    bg.add_task(run_translate, req, jid)
    return {"job_id": jid}

@app.post("/api/tts")
def tts(req: TTSReq, bg: BackgroundTasks):
    jid = uuid.uuid4().hex
    bg.add_task(run_tts, req, jid)
    return {"job_id": jid}

@app.get("/api/job/{jid}")
def job(jid: str):
    return JOBS.get(jid, {"status": "unknown", "progress": 0})

@app.get("/")
def index():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print(" ShortForge AI запущен!")
    print("="*60)
    print(f"\n  Открой в браузере: http://127.0.0.1:8765\n")
    print(f"  Файлы сохраняются в: {OUT}\n")
    uvicorn.run(app, host="127.0.0.1", port=8765)
