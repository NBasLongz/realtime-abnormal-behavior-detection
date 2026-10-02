import { useMemo, useState } from 'react';
import {
  ArrowLeft,
  BarChart3,
  Check,
  Clock,
  Download,
  FileVideo,
  FolderOpen,
  Gauge,
  HardDrive,
  Layers,
  Monitor,
  Play,
  Sparkles,
  UploadCloud,
} from 'lucide-react';

const STEPS = [
  { name: 'Tải tệp', pct: '30%' },
  { name: 'Trích xuất khung xương', pct: '60%' },
  { name: 'Phân loại hành vi', pct: '90%' },
  { name: 'Hoàn tất', pct: '100%' },
];

const BEHAVIOR_BADGE = {
  walking: 'bg-blue-50 text-blue-600 border-blue-200',
  standing: 'bg-amber-50 text-amber-700 border-amber-200',
  running: 'bg-red-50 text-red-600 border-red-200',
  falling: 'bg-amber-50 text-amber-700 border-amber-200',
  fighting: 'bg-red-50 text-red-600 border-red-200',
  loitering: 'bg-indigo-50 text-indigo-600 border-indigo-200',
};

const DOT_COLOR = {
  walking: '#2563EB',
  standing: '#F59E0B',
  running: '#EF4444',
  falling: '#F59E0B',
  fighting: '#EF4444',
  loitering: '#6366F1',
};

const DEFAULT_SAMPLES = [
  { time: '00:02.4', behavior: 'walking', person: 'person_1' },
  { time: '00:15.6', behavior: 'walking', person: 'person_5' },
  { time: '00:05.7', behavior: 'standing', person: 'person_2' },
  { time: '00:18.2', behavior: 'running', person: 'person_6' },
  { time: '00:08.9', behavior: 'walking', person: 'person_3' },
  { time: '00:22.7', behavior: 'walking', person: 'person_7' },
  { time: '00:12.3', behavior: 'walking', person: 'person_4' },
  { time: '00:28.1', behavior: 'standing', person: 'person_8' },
];

export default function AnalyzeVideo({ onAnalyzed }) {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('surveillance_01.mp4');
  const [fileSize, setFileSize] = useState('24.7 MB');
  const [progress, setProgress] = useState(90);
  const [running, setRunning] = useState(false);
  const [analysisResult, setAnalysisResult] = useState({
    duration: '32.4 s',
    resolution: '1920 x 1080',
    fps: '30.0',
    frames: '972',
  });
  const [timeline, setTimeline] = useState(DEFAULT_SAMPLES);

  const rawVideoSrc = useMemo(() => {
    if (file) return URL.createObjectURL(file);
    return '/surveillance_01.mp4';
  }, [file]);

  const analyzedVideoSrc = useMemo(() => {
    return '/analyzed_surveillance_01.mp4';
  }, []);

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setFileName(f.name);
      setFileSize(`${(f.size / (1024 * 1024)).toFixed(1)} MB`);
      setProgress(10);
    }
  };

  const handleStartAnalysis = async () => {
    setRunning(true);
    setProgress(30);

    const timer = setInterval(() => {
      setProgress((p) => (p < 90 ? p + 20 : p));
    }, 400);

    try {
      if (file) {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('/api/video/process', { method: 'POST', body: formData });
        if (res.ok) {
          const data = await res.json();
          clearInterval(timer);
          setProgress(100);
          setRunning(false);
          if (data.info) {
            setAnalysisResult({
              duration: typeof data.info.duration === 'string' ? data.info.duration : `${data.info.duration} s`,
              resolution: data.info.resolution,
              fps: String(data.info.fps),
              frames: String(data.info.frames),
            });
          }
          if (data.timeline && data.timeline.length > 0) {
            const mapped = data.timeline.map(([t, b, p]) => ({ time: t, behavior: b, person: p }));
            setTimeline(mapped);
          }
          if (onAnalyzed) {
            onAnalyzed({
              id: Date.now(),
              time: Date.now(),
              file: fileName,
              duration: parseFloat(data.info.duration) || 32.4,
              resolution: data.info.resolution || '1920x1080',
              fps: parseFloat(data.info.fps) || 30.0,
              frames: parseInt(data.info.frames) || 972,
              output_url: data.output_url,
            });
          }
          return;
        }
      }
    } catch {
      // Fallback
    }

    setTimeout(() => {
      clearInterval(timer);
      setProgress(100);
      setRunning(false);
    }, 1200);
  };

  return (
    <div className="space-y-6">
      {/* Page Header with Arrow Back */}
      <div className="flex items-center gap-3">
        <button
          className="grid size-9 place-items-center rounded-lg border border-slate-200 bg-white text-slate-600 shadow-xs hover:bg-slate-50 transition-colors"
          aria-label="Back"
        >
          <ArrowLeft size={18} />
        </button>
        <div>
          <h1 className="text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">
            Analyze Video
          </h1>
          <p className="text-xs text-slate-500 sm:text-sm">
            Upload a video and let AI detect and analyze behaviors
          </p>
        </div>
      </div>

      {/* Main Grid: 2/5 Left, 3/5 Right */}
      <div className="grid gap-6 lg:grid-cols-5">
        {/* Left Column (2/5): Upload & Live Analysis */}
        <div className="space-y-5 lg:col-span-2">
          {/* Upload Card */}
          <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-xs">
            <label className="group flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-blue-200 bg-blue-50/20 py-8 px-4 text-center transition-all hover:border-blue-400 hover:bg-blue-50/40">
              <UploadCloud size={44} className="text-blue-500 transition-transform group-hover:scale-105" />
              <div className="mt-3 text-base font-bold text-slate-800">Upload Video</div>
              <div className="mt-1 text-xs text-slate-500">
                Drag and drop your video file here, or click to browse
              </div>
              <div className="mt-1.5 text-[11px] text-slate-400">
                Supported formats: MP4, AVI, MOV, MKV &nbsp;|&nbsp; Max size: 500MB
              </div>
              <input
                type="file"
                accept=".mp4,.avi,.mov,.mkv,video/*"
                onChange={handleFileChange}
                className="hidden"
              />
              <span className="mt-4 inline-flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2 text-sm font-semibold text-white shadow-xs transition-colors hover:bg-blue-700">
                <FolderOpen size={16} /> Choose File
              </span>
            </label>
          </div>

          {/* Video Preview Card */}
          <div className="rounded-2xl border border-slate-200/80 bg-white p-4 shadow-xs space-y-3.5">
            <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-slate-950">
              <span className="absolute left-3 top-3 z-10 flex items-center gap-1.5 rounded bg-black/60 px-2.5 py-1 text-xs font-medium text-white backdrop-blur-sm border border-white/10">
                <FileVideo size={13} /> {fileName}
              </span>
              <video
                src={rawVideoSrc}
                controls
                className="size-full object-cover"
              />
            </div>

            {/* Video File Specs */}
            <div className="grid grid-cols-2 gap-3">
              <div className="flex items-center gap-2.5 rounded-lg border border-slate-200/80 bg-slate-50 px-3.5 py-2.5">
                <FileVideo size={16} className="text-blue-600 shrink-0" />
                <div className="min-w-0">
                  <div className="text-[11px] text-slate-400 font-medium leading-none">Name</div>
                  <div className="truncate text-xs font-semibold text-slate-800 mt-0.5">{fileName}</div>
                </div>
              </div>
              <div className="flex items-center gap-2.5 rounded-lg border border-slate-200/80 bg-slate-50 px-3.5 py-2.5">
                <HardDrive size={16} className="text-blue-600 shrink-0" />
                <div className="min-w-0">
                  <div className="text-[11px] text-slate-400 font-medium leading-none">Size</div>
                  <div className="truncate text-xs font-semibold text-slate-800 mt-0.5">{fileSize}</div>
                </div>
              </div>
            </div>

            {/* Analyze CTA Button */}
            <button
              onClick={handleStartAnalysis}
              disabled={running}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-emerald-400 to-teal-500 py-3.5 text-base font-bold text-white shadow-md shadow-emerald-500/25 transition-all hover:from-emerald-500 hover:to-teal-600 active:scale-[0.99] disabled:opacity-50"
            >
              <Play size={18} fill="currentColor" /> {running ? 'Analyzing AI...' : 'Analyze'}
            </button>
          </div>

          {/* Analysis Progress & Celebration Banner */}
          <div className="rounded-2xl border border-slate-200/80 bg-white p-4.5 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-bold text-slate-800">Analysis Progress</span>
              <span className="text-sm font-bold text-blue-600">{progress}%</span>
            </div>

            {/* Stepper with Connecting Line */}
            <div className="relative flex items-center justify-between px-2 pt-1">
              <div className="absolute left-6 right-6 top-4 -translate-y-1/2 h-1 bg-slate-100 -z-0">
                <div
                  className="h-full bg-emerald-500 transition-all duration-300"
                  style={{ width: `${Math.min(100, (progress / 100) * 100)}%` }}
                />
              </div>

              {STEPS.map((s, i) => {
                const targetPct = (i + 1) * 25;
                const isPassed = progress >= targetPct || (i === 3 && progress >= 90);
                return (
                  <div key={s.name} className="relative z-10 flex flex-col items-center text-center">
                    <span
                      className={`grid size-7 place-items-center rounded-full text-xs font-bold transition-all shadow-xs ${
                        isPassed
                          ? 'bg-emerald-500 text-white'
                          : i === 3
                          ? 'bg-blue-600 text-white'
                          : 'bg-slate-200 text-slate-500'
                      }`}
                    >
                      <Check size={14} />
                    </span>
                    <span className="mt-2 text-[11px] font-semibold text-slate-700 max-w-[70px] leading-tight">
                      {s.name}
                    </span>
                    <span className="text-[10px] text-slate-400 mt-0.5">{s.pct}</span>
                  </div>
                );
              })}
            </div>

            {/* Celebration Mint Banner */}
            <div className="flex items-center justify-between rounded-xl border border-emerald-200/90 bg-emerald-50/80 px-4 py-3">
              <div className="flex items-center gap-2.5">
                <span className="text-xl">🎉</span>
                <div>
                  <div className="text-xs font-bold text-emerald-900">Phân tích hoàn tất!</div>
                  <div className="text-[11px] text-emerald-700">Video của bạn đã được xử lý thành công.</div>
                </div>
              </div>
              <div className="flex items-center gap-1 text-lg">
                <span>🎈</span>
                <span>🎈</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column (3/5): Analysis Results */}
        <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-xs space-y-5 lg:col-span-3">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h2 className="flex items-center gap-2 text-lg font-bold text-slate-900">
              <BarChart3 size={20} className="text-blue-600" />
              Analysis Results
            </h2>
            <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-600">
              ✓ Completed
            </span>
          </div>

          {/* 4 Top Metric Cards (Compact Height) */}
          <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
            {/* Duration */}
            <div className="flex items-center gap-2.5 rounded-xl border border-slate-200/90 bg-white px-3 py-2 shadow-xs">
              <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-sky-50 text-sky-600">
                <Clock size={16} />
              </span>
              <div className="min-w-0">
                <div className="text-[11px] font-medium text-slate-500 leading-tight">Duration</div>
                <div className="text-base font-bold text-slate-900 leading-none mt-0.5 truncate">{analysisResult.duration}</div>
              </div>
            </div>

            {/* Resolution */}
            <div className="flex items-center gap-2.5 rounded-xl border border-slate-200/90 bg-white px-3 py-2 shadow-xs">
              <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-purple-50 text-purple-600">
                <Monitor size={16} />
              </span>
              <div className="min-w-0">
                <div className="text-[11px] font-medium text-slate-500 leading-tight">Resolution</div>
                <div className="text-base font-bold text-slate-900 leading-none mt-0.5 truncate">{analysisResult.resolution}</div>
              </div>
            </div>

            {/* FPS */}
            <div className="flex items-center gap-2.5 rounded-xl border border-slate-200/90 bg-white px-3 py-2 shadow-xs">
              <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-emerald-50 text-emerald-600">
                <Gauge size={16} />
              </span>
              <div className="min-w-0">
                <div className="text-[11px] font-medium text-slate-500 leading-tight">FPS</div>
                <div className="text-base font-bold text-slate-900 leading-none mt-0.5 truncate">{analysisResult.fps}</div>
              </div>
            </div>

            {/* Frames */}
            <div className="flex items-center gap-2.5 rounded-xl border border-slate-200/90 bg-white px-3 py-2 shadow-xs">
              <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-pink-50 text-pink-600">
                <Layers size={16} />
              </span>
              <div className="min-w-0">
                <div className="text-[11px] font-medium text-slate-500 leading-tight">Frames</div>
                <div className="text-base font-bold text-slate-900 leading-none mt-0.5 truncate">{analysisResult.frames}</div>
              </div>
            </div>
          </div>

          {/* Video with Detection Results */}
          <div className="space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800">
              <Sparkles size={14} className="text-purple-600" /> Video with Detection Results
            </div>
            <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-slate-950">
              <span className="absolute right-3 top-3 z-10 flex items-center gap-1.5 rounded bg-black/70 px-2 py-0.5 text-[11px] font-semibold text-white backdrop-blur-sm border border-white/10">
                YOLOv8 + BiLSTM
              </span>
              <video
                src={analyzedVideoSrc}
                controls
                className="size-full object-cover"
              />
            </div>
          </div>

          {/* Detected Behaviors (Sample) */}
          <div className="space-y-2.5">
            <div className="text-xs font-bold text-slate-800">
              Detected Behaviors (Sample)
            </div>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {timeline.slice(0, 8).map((item, idx) => {
                const badge = BEHAVIOR_BADGE[item.behavior] || 'bg-slate-50 text-slate-700 border-slate-200';
                const dot = DOT_COLOR[item.behavior] || '#2563EB';
                return (
                  <div
                    key={idx}
                    className="flex items-center justify-between rounded-lg border border-slate-200/90 bg-white px-3 py-2 text-xs shadow-xs"
                  >
                    <div className="flex items-center gap-2">
                      <span className="size-2 rounded-full" style={{ backgroundColor: dot }} />
                      <span className="font-mono text-slate-500">{item.time}</span>
                    </div>
                    <span className={`rounded-md border px-2.5 py-0.5 font-semibold capitalize ${badge}`}>
                      {item.behavior}
                    </span>
                    <span className="font-mono text-slate-400">({item.person})</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Download Result Button */}
          <a
            href={analyzedVideoSrc}
            download="analyzed_result.mp4"
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-500 via-blue-600 to-indigo-600 py-3.5 text-sm font-semibold text-white shadow-md shadow-blue-500/25 transition-all hover:from-blue-600 hover:to-indigo-700 active:scale-[0.99]"
          >
            <Download size={18} /> Download Result
          </a>
        </div>
      </div>
    </div>
  );
}
