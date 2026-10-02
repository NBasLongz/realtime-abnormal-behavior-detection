import { useEffect, useState } from 'react';
import { Bell, Search, Sun } from 'lucide-react';
import Sidebar, { TABS } from '../components/Sidebar';
import { seedVideos } from '../data';
import AnalyzeVideo from '../tabs/AnalyzeVideo';
import BatchProcessing from '../tabs/BatchProcessing';
import Dashboard from '../tabs/Dashboard';
import HistoryTab from '../tabs/HistoryTab';

export default function AdminApp() {
  const [tab, setTab] = useState('analyze');
  const [videos, setVideos] = useState(seedVideos);

  useEffect(() => {
    fetch('/api/videos')
      .then((r) => {
        if (!r.ok) throw new Error('API status ' + r.status);
        return r.json();
      })
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setVideos(data);
        }
      })
      .catch((err) => {
        console.warn('Backend videos API not reachable, using default data:', err);
      });
  }, []);

  const handleNewVideo = (newRecord) => {
    setVideos((prev) => [newRecord, ...prev]);
  };

  const handleClearHistory = () => {
    fetch('/api/videos', { method: 'DELETE' }).catch(() => {});
    setVideos([]);
  };

  const content = {
    analyze: <AnalyzeVideo onAnalyzed={handleNewVideo} />,
    batch: <BatchProcessing />,
    dashboard: <Dashboard videos={videos} />,
    history: <HistoryTab videos={videos} onClear={handleClearHistory} />,
  }[tab];

  return (
    <div className="flex min-h-screen bg-[#F0F4FA]">
      {/* Deep Navy Left Sidebar */}
      <Sidebar tab={tab} setTab={setTab} videos={videos} />

      {/* Main Container */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Deep Navy Top App Bar - Matching Screenshot */}
        <header className="relative flex items-center justify-between border-b border-[#1A285A] bg-[#0B1437] px-6 py-3.5 text-white shadow-md">
          {/* Subtle top right ambient flare */}
          <div className="pointer-events-none absolute right-0 top-0 h-full w-96 bg-gradient-to-l from-blue-600/15 to-transparent" />

          {/* Search Box */}
          <div className="flex items-center gap-3">
            <label className="flex w-80 items-center gap-2.5 rounded-full border border-blue-400/20 bg-[#142354] px-4 py-2 text-sm text-slate-300 shadow-inner focus-within:border-blue-400 transition-colors sm:w-96">
              <Search size={16} className="text-slate-400" />
              <input
                placeholder="Search videos, files, or analysis..."
                className="w-full bg-transparent text-sm text-white placeholder-slate-400 outline-none"
              />
            </label>
          </div>

          {/* Top Right Controls & User Profile */}
          <div className="flex items-center gap-4">
            {/* Notification Bell */}
            <button
              className="relative grid size-8 place-items-center rounded-lg text-slate-300 hover:text-white transition-colors"
              aria-label="Notifications"
            >
              <Bell size={18} />
              <span className="absolute right-1 top-1 size-2 rounded-full bg-rose-500" />
            </button>

            {/* Sun / Theme Icon */}
            <button
              className="grid size-8 place-items-center rounded-lg text-slate-300 hover:text-white transition-colors"
              aria-label="Toggle Theme"
            >
              <Sun size={18} />
            </button>

            {/* User Profile */}
            <div className="flex items-center gap-3 pl-2 border-l border-white/10">
              <div className="size-9 rounded-full bg-gradient-to-tr from-blue-500 to-indigo-600 p-[1.5px]">
                <div className="grid size-full place-items-center rounded-full bg-[#0B1437] text-xs font-bold text-white">
                  NL
                </div>
              </div>
              <div className="hidden text-left text-sm leading-tight sm:block">
                <div className="font-semibold text-white">Nguyen Ba Long</div>
                <div className="text-[11px] text-slate-400">Admin</div>
              </div>
            </div>
          </div>
        </header>

        {/* Content Canvas */}
        <main className="min-w-0 flex-1 p-5 md:p-6 lg:p-8">
          {content}
        </main>
      </div>
    </div>
  );
}
