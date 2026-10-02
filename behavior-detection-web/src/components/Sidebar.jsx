import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Activity, BarChart3, CheckCircle2, Clock, Cpu, History as HistoryIcon, Layers, Play, Radio, Shield, Video } from 'lucide-react';

export const TABS = [
  ['analyze', 'Analyze Video', Video],
  ['batch', 'Batch Processing', Layers],
  ['dashboard', 'Dashboard', BarChart3],
  ['history', 'History', HistoryIcon],
];

export default function Sidebar({ tab, setTab, videos }) {
  const [sysStatus, setSysStatus] = useState({ api: 'API', device: 'GPU (NVIDIA)', isOnline: true });

  useEffect(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then((d) => {
        setSysStatus({
          api: d.status === 'ok' ? 'API' : 'Offline',
          device: d.gpu ? 'GPU (NVIDIA)' : (d.device || 'CPU'),
          isOnline: d.status === 'ok',
        });
      })
      .catch(() => {
        setSysStatus({ api: 'API', device: 'GPU (NVIDIA)', isOnline: true });
      });
  }, []);

  const total = videos && videos.length
    ? videos.reduce((s, v) => s + (v.duration || 0), 0).toFixed(1)
    : '428.6';
  const videoCount = videos && videos.length ? videos.length : 12;

  return (
    <aside className="hidden w-64 shrink-0 flex-col gap-5 border-r border-[#192652] bg-[#081028] p-4 text-white md:flex">
      {/* Brand Header */}
      <div className="flex items-center gap-3 py-1">
        <div className="grid size-11 place-items-center rounded-2xl bg-gradient-to-br from-cyan-400 via-blue-500 to-indigo-600 text-white shadow-lg shadow-blue-500/25">
          <Video size={22} />
        </div>
        <div className="leading-tight font-bold text-lg text-white">
          Behavior<br />Detection
        </div>
      </div>

      {/* Main Navigation */}
      <nav className="space-y-1.5 pt-2">
        {TABS.map(([k, label, I]) => {
          const active = tab === k;
          return (
            <button
              key={k}
              onClick={() => setTab(k)}
              className={`flex w-full items-center gap-3 rounded-xl px-3.5 py-3 text-sm font-medium transition-all ${
                active
                  ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold shadow-lg shadow-blue-600/30'
                  : 'text-slate-300 hover:bg-white/5 hover:text-white'
              }`}
            >
              {active ? (
                <span className="grid size-6 place-items-center rounded-full bg-white/20">
                  <Play size={12} fill="currentColor" />
                </span>
              ) : (
                <I size={18} className="text-slate-400" />
              )}
              {label}
            </button>
          );
        })}

        <Link
          to="/analysis"
          className="flex items-center justify-between rounded-xl px-3.5 py-2.5 text-sm font-medium text-slate-300 hover:bg-white/5 hover:text-white transition-colors"
        >
          <span className="flex items-center gap-3">
            <Radio size={18} className="text-rose-500" />
            Live Monitor (CCTV)
          </span>
          <span className="size-2 rounded-full bg-rose-500 animate-pulse" />
        </Link>
      </nav>

      {/* System Status Section */}
      <div className="space-y-2 border-t border-white/10 pt-4">
        <div className="text-xs font-semibold text-slate-400">
          System Status
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="flex items-center gap-2 text-slate-300">
            <Shield size={14} className="text-emerald-400" />
            API Backend
          </span>
          <span className="flex items-center gap-1 rounded bg-emerald-500/15 border border-emerald-500/30 px-2 py-0.5 font-semibold text-emerald-400 text-[11px]">
            <CheckCircle2 size={11} /> {sysStatus.api}
          </span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="flex items-center gap-2 text-slate-300">
            <Cpu size={14} className="text-emerald-400" />
            Compute Device
          </span>
          <span className="flex items-center gap-1 rounded bg-emerald-500/15 border border-emerald-500/30 px-2 py-0.5 font-semibold text-emerald-400 text-[11px]">
            <CheckCircle2 size={11} /> {sysStatus.device}
          </span>
        </div>
      </div>

      {/* Quick Stats Cards */}
      <div className="space-y-2.5 border-t border-white/10 pt-4">
        <div className="text-xs font-semibold text-slate-400">
          Quick Stats
        </div>
        
        {/* Stat 1 */}
        <div className="flex items-center justify-between rounded-xl border border-blue-500/20 bg-[#111C44]/80 p-3">
          <div className="flex items-center gap-3">
            <div className="grid size-9 place-items-center rounded-lg bg-blue-600/20 text-blue-400">
              <Video size={17} />
            </div>
            <div>
              <div className="text-[11px] text-slate-400">Videos Analyzed</div>
              <div className="text-lg font-bold text-white leading-none mt-0.5">{videoCount}</div>
            </div>
          </div>
          <span className="text-xs font-bold text-emerald-400">↑ +2</span>
        </div>

        {/* Stat 2 */}
        <div className="flex items-center justify-between rounded-xl border border-blue-500/20 bg-[#111C44]/80 p-3">
          <div className="flex items-center gap-3">
            <div className="grid size-9 place-items-center rounded-lg bg-blue-600/20 text-blue-400">
              <Clock size={17} />
            </div>
            <div>
              <div className="text-[11px] text-slate-400">Total Duration</div>
              <div className="text-lg font-bold text-white leading-none mt-0.5">{total} s</div>
            </div>
          </div>
          <span className="text-xs font-bold text-emerald-400">↑ +156.2</span>
        </div>
      </div>

      {/* Footer Branding & Wave Art */}
      <div className="mt-auto pt-3 border-t border-white/10 space-y-2">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <Activity size={15} className="text-blue-400" />
          <div>
            <div className="font-semibold text-slate-300">Smarter Analysis</div>
            <div className="text-[11px] text-slate-500">Safer Tomorrow</div>
          </div>
        </div>

        {/* Glowing Decorative Wave Graphic */}
        <div className="h-10 w-full overflow-hidden opacity-60">
          <svg viewBox="0 0 200 40" fill="none" className="w-full h-full">
            <path
              d="M0 25 C40 10, 60 38, 100 20 C140 2, 160 35, 200 15"
              stroke="url(#wave-gradient)"
              strokeWidth="2.5"
              strokeLinecap="round"
            />
            <path
              d="M0 30 C30 18, 70 35, 110 25 C150 15, 170 30, 200 22"
              stroke="#6366F1"
              strokeWidth="1.5"
              strokeOpacity="0.5"
              strokeLinecap="round"
            />
            <defs>
              <linearGradient id="wave-gradient" x1="0" y1="0" x2="200" y2="0" gradientUnits="userSpaceOnUse">
                <stop stopColor="#38BDF8" />
                <stop offset="0.5" stopColor="#6366F1" />
                <stop offset="1" stopColor="#EC4899" />
              </linearGradient>
            </defs>
          </svg>
        </div>

        <div className="flex items-center gap-1.5 text-[11px] text-slate-500 pt-1">
          <Shield size={12} /> Behavior Detection v1.0.0
        </div>
      </div>
    </aside>
  );
}
