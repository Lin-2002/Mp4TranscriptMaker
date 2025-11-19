import os, re, shutil, threading, traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from pydub import AudioSegment
from faster_whisper import WhisperModel

APP_TITLE = "MP4 逐字稿轉寫（GUI）"

# ---------- 核心處理邏輯 ----------

def ensure_ffmpeg(base_dir: Path):
    """找 ffmpeg：優先同資料夾 ffmpeg.exe，其次 PATH。找到後設定給 pydub 使用。"""
    local_ffmpeg = base_dir / "ffmpeg.exe"
    ff = None
    if local_ffmpeg.exists():
        ff = str(local_ffmpeg)
    else:
        ff = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not ff:
        raise RuntimeError("找不到 ffmpeg，請將 ffmpeg.exe 放在本程式同資料夾，或把 ffmpeg 加到 PATH。")
    AudioSegment.converter = ff
    return ff

def extract_wav_from_mp4(mp4_path: Path, log):
    wav = mp4_path.with_suffix(".wav")
    log(f"抽音中：{mp4_path.name} → {wav.name}")
    AudioSegment.from_file(mp4_path).export(wav, format="wav")
    return wav

def chunk_wav(wav_path: Path, minutes: int, log):
    audio = AudioSegment.from_wav(wav_path)
    step = minutes * 60 * 1000  # ms
    parts = []
    for i in range(0, len(audio), step):
        seg = audio[i:i+step]
        out = wav_path.parent / f"{wav_path.stem}.part{i//step:03d}.wav"
        # 轉 16k / mono，利於 ASR 穩定與資源使用
        seg.export(out, format="wav", parameters=["-ar","16000","-ac","1"])
        parts.append(out)
    log(f"切段完成，共 {len(parts)} 段（每段約 {minutes} 分鐘）")
    return parts

def paragraphize(txt: str, target_max=170):
    sents = re.split(r"([。！？；])", txt)
    sents = ["".join(sents[i:i+2]).strip() for i in range(0, len(sents), 2)]
    buf, cur = [], ""
    for s in sents:
        if len(cur) + len(s) < target_max:
            cur += s
        else:
            if cur.strip():
                buf.append(cur.strip())
            cur = s
    if cur.strip():
        buf.append(cur.strip())
    return buf

# ---------- GUI App ----------

class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=12)
        self.master.title(APP_TITLE)
        self.master.geometry("820x620")

        # 狀態
        self.running = False
        self.worker = None
        self.base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

        # 變數
        self.mp4_var = tk.StringVar(value="")
        self.minutes_var = tk.IntVar(value=20)
        self.model_var = tk.StringVar(value="large-v3")  # 可改 medium / small
        self.language_var = tk.StringVar(value="zh")     # zh / en / ja / auto
        self.vad_ms_var = tk.IntVar(value=500)
        self.beam_var = tk.IntVar(value=5)
        self.cond_prev_var = tk.BooleanVar(value=False)
        self.paragraph_max_var = tk.IntVar(value=170)

        # 介面
        self._build_ui()

    def _build_ui(self):
        # 第一列：mp4 檔名/路徑（置頂）
        row = 0
        ttk.Label(self, text="MP4 檔名或路徑：").grid(row=row, column=0, sticky="e", padx=6, pady=6)
        e = ttk.Entry(self, textvariable=self.mp4_var, width=70)
        e.grid(row=row, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(self, text="瀏覽…", command=self.on_browse).grid(row=row, column=2, padx=6, pady=6)

        # 參數列
        row += 1
        frm = ttk.Frame(self); frm.grid(row=row, column=0, columnspan=3, sticky="we", pady=(4, 8))
        frm.columnconfigure(0, weight=1); frm.columnconfigure(1, weight=1); frm.columnconfigure(2, weight=1); frm.columnconfigure(3, weight=1)

        # 切段分鐘
        ttk.Label(frm, text="每段（分鐘）").grid(row=0, column=0, sticky="w", padx=6)
        ttk.Spinbox(frm, from_=5, to=60, textvariable=self.minutes_var, width=8).grid(row=1, column=0, sticky="w", padx=6)

        # 模型
        ttk.Label(frm, text="Whisper 模型").grid(row=0, column=1, sticky="w", padx=6)
        cb = ttk.Combobox(frm, textvariable=self.model_var, values=["large-v3", "medium", "small"], state="readonly", width=12)
        cb.grid(row=1, column=1, sticky="w", padx=6)

        # 語言
        ttk.Label(frm, text="語言").grid(row=0, column=2, sticky="w", padx=6)
        ttk.Combobox(frm, textvariable=self.language_var, values=["zh", "en", "ja", "auto"], state="readonly", width=8).grid(row=1, column=2, sticky="w", padx=6)

        # Beam size
        ttk.Label(frm, text="Beam size").grid(row=0, column=3, sticky="w", padx=6)
        ttk.Spinbox(frm, from_=1, to=10, textvariable=self.beam_var, width=8).grid(row=1, column=3, sticky="w", padx=6)

        # 第二排參數
        row2 = 2
        ttk.Label(frm, text="VAD 靜默(ms)").grid(row=row2, column=0, sticky="w", padx=6, pady=(8,0))
        ttk.Spinbox(frm, from_=100, to=2000, increment=50, textvariable=self.vad_ms_var, width=8).grid(row=row2+1, column=0, sticky="w", padx=6)

        ttk.Label(frm, text="段落字數上限").grid(row=row2, column=1, sticky="w", padx=6, pady=(8,0))
        ttk.Spinbox(frm, from_=120, to=240, textvariable=self.paragraph_max_var, width=8).grid(row=row2+1, column=1, sticky="w", padx=6)

        ttk.Checkbutton(frm, text="不依賴上一段文字（condition_on_previous_text=False）", variable=self.cond_prev_var)\
            .grid(row=row2+1, column=2, columnspan=2, sticky="w", padx=6)

        # 控制列
        row += 1
        ctl = ttk.Frame(self); ctl.grid(row=row, column=0, columnspan=3, sticky="we", pady=(4, 8))
        ttk.Button(ctl, text="開始轉寫", command=self.on_start).pack(side="left", padx=4)
        ttk.Button(ctl, text="開啟輸出資料夾", command=self.open_outdir).pack(side="left", padx=4)
        self.progress = ttk.Progressbar(ctl, length=320, mode="determinate")
        self.progress.pack(side="right", padx=4)

        # 日誌區
        row += 1
        self.logbox = tk.Text(self, height=20, wrap="word")
        self.logbox.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(4, 0))
        self.columnconfigure(1, weight=1)
        self.rowconfigure(row, weight=1)

        self.pack(fill="both", expand=True)

    # ---------- 事件處理 ----------

    def on_browse(self):
        path = filedialog.askopenfilename(
            title="選擇 MP4 檔案",
            initialdir=self.base_dir,
            filetypes=[("MP4 檔案", "*.mp4"), ("所有檔案", "*.*")]
        )
        if path:
            self.mp4_var.set(path)

    def open_outdir(self):
        outdir = self.base_dir / "transcripts"
        outdir.mkdir(exist_ok=True)
        try:
            os.startfile(outdir)  # Windows
        except Exception:
            messagebox.showinfo("輸出資料夾", f"路徑：{outdir}")

    def on_start(self):
        if self.running:
            return
        mp4_input = self.mp4_var.get().strip()
        if not mp4_input:
            messagebox.showwarning("請選擇檔案", "請先輸入或選擇要轉寫的 MP4 檔案。")
            return

        # 解析路徑：可填「同資料夾的檔名」或完整路徑
        mp4_path = Path(mp4_input)
        if not mp4_path.is_absolute():
            # 優先同資料夾
            mp4_path = self.base_dir / mp4_path.name
        if not mp4_path.exists():
            messagebox.showerror("找不到檔案", f"無法找到：{mp4_path}")
            return

        # 讀取參數
        params = dict(
            mp4_path=mp4_path,
            minutes=int(self.minutes_var.get()),
            model=self.model_var.get(),
            language=self.language_var.get(),
            vad_ms=int(self.vad_ms_var.get()),
            beam=int(self.beam_var.get()),
            cond_prev=bool(self.cond_prev_var.get()),
            para_max=int(self.paragraph_max_var.get()),
        )

        self.running = True
        self.progress["value"] = 0
        self.logbox.delete("1.0", "end")
        self.log("開始轉寫…")
        self.worker = threading.Thread(target=self.run_pipeline, args=(params,))
        self.worker.daemon = True
        self.worker.start()

    def run_pipeline(self, p):
        try:
            ensure_ffmpeg(self.base_dir)
            self.log(f"檢查 ffmpeg 完成。")

            # 1) 抽音
            wav = extract_wav_from_mp4(p["mp4_path"], self.log)

            # 2) 分段
            parts = chunk_wav(wav, minutes=p["minutes"], log=self.log)
            self.progress.configure(maximum=len(parts), value=0)

            # 3) 轉寫
            lang = p["language"]
            use_lang = None if (not lang or lang.lower() == "auto") else lang
            self.log(f"載入模型：{p['model']}（首次載入可能較久）")
            model = WhisperModel(p["model"])

            texts = []
            for idx, part in enumerate(parts, 1):
                self.log(f"[{idx}/{len(parts)}] 轉寫：{part.name}")
                segs, info = model.transcribe(
                    str(part),
                    language=use_lang,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=p["vad_ms"]),
                    beam_size=p["beam"],
                    condition_on_previous_text=p["cond_prev"]
                )
                piece = "".join(s.text for s in segs)
                texts.append(piece)
                self.progress_step()

            full_txt = "".join(texts)
            self.log(f"轉寫完成，字數：約 {len(full_txt)}")

            # 4) 段落化
            paras = paragraphize(full_txt, target_max=p["para_max"])

            # 5) 輸出
            outdir = self.base_dir / "transcripts"; outdir.mkdir(exist_ok=True)
            outfile = outdir / f"{p['mp4_path'].stem}.txt"
            outfile.write_text("\n\n".join(paras), encoding="utf-8")
            self.log(f"完成 → {outfile}")

            self.log("✅ 全部完成！可點「開啟輸出資料夾」查看檔案。")
        except Exception as e:
            self.log("❌ 發生錯誤：\n" + "".join(traceback.format_exception_only(type(e), e)))
            self.log(traceback.format_exc())
            messagebox.showerror("錯誤", str(e))
        finally:
            self.running = False

    # ---------- UI 輔助 ----------

    def log(self, msg: str):
        self.logbox.insert("end", str(msg).rstrip() + "\n")
        self.logbox.see("end")
        self.update_idletasks()

    def progress_step(self):
        self.progress["value"] = self.progress["value"] + 1
        self.update_idletasks()

# ---------- 進入點 ----------
if __name__ == "__main__":
    root = tk.Tk()
    # Windows 11 預設主題可能較素，這裡用 ttk 樣式微調（可略）
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    style = ttk.Style()
    # 如果系統支援 "vista" 或 "clam" 主題可套用
    for theme in ("vista", "clam", "default"):
        try:
            style.theme_use(theme)
            break
        except Exception:
            continue
    App(root)
    root.mainloop()
