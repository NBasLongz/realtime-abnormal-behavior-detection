import { useState } from 'react';
import { Film, FileDown, Layers, PlaySquare, Settings2, Timer, Trash2 } from 'lucide-react';
import { Card, PageHead } from '../ui';
import { fmtDateTime, RESOLUTION_BADGE, RESOLUTION_COLOR } from '../data';

const SORTS = {
  newest: ['Ngày (Mới nhất)', (a, b) => b.time - a.time],
  oldest: ['Ngày (Cũ nhất)', (a, b) => a.time - b.time],
  duration: ['Thời lượng', (a, b) => b.duration - a.duration],
};

const STAT_THEMES = [
  { label: 'Tổng số video', key: 'n', icon: Film, color: '#6366F1' },
  { label: 'Tổng thời lượng', key: 'dur', icon: Timer, color: '#0284C7' },
  { label: 'Thời lượng TB', key: 'avgDur', icon: PlaySquare, color: '#0D9488' },
  { label: 'FPS Trung bình', key: 'fps', icon: Settings2, color: '#10B981' },
  { label: 'Độ phân giải phổ biến', key: 'res', icon: Layers, color: '#F59E0B' },
  { label: 'Tổng số khung hình', key: 'frames', icon: Layers, color: '#8B5CF6' },
];

export default function HistoryTab({ videos, onClear }) {
  const [sort, setSort] = useState('newest');
  const [limit, setLimit] = useState(10);
  const shown = [...videos].sort(SORTS[sort][1]).slice(0, limit);

  const n = videos.length;
  const totalDur = videos.reduce((s, v) => s + v.duration, 0);
  const frames = videos.reduce((s, v) => s + v.frames, 0);
  const counts = {};
  videos.forEach((v) => {
    counts[v.resolution] = (counts[v.resolution] || 0) + 1;
  });
  const common = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? '--';

  const statValues = [
    n,
    `${totalDur.toFixed(1)}s`,
    `${n ? (totalDur / n).toFixed(1) : 0}s`,
    n ? (videos.reduce((s, v) => s + v.fps, 0) / n).toFixed(1) : 0,
    common,
    frames.toLocaleString('vi-VN'),
  ];

  const exportCsv = () => {
    const rows = [
      ['Time', 'File', 'Duration', 'Resolution', 'FPS', 'Frames'],
      ...videos.map((v) => [
        fmtDateTime(v.time),
        v.file,
        v.duration,
        v.resolution,
        v.fps,
        v.frames,
      ]),
    ];
    const csv = rows.map((r) => r.map((c) => `"${c}"`).join(',')).join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8' }));
    a.download = 'history.csv';
    a.click();
  };

  return (
    <>
      <PageHead
        title="History"
        sub="Nhật ký toàn bộ các phiên phân tích video, tra cứu thông tin và xuất báo cáo CSV."
      />
      <Card>
        {/* Controls */}
        <div className="mb-5 flex flex-wrap items-end gap-6 text-sm">
          <label className="flex flex-col gap-1.5">
            <span className="text-xs font-semibold text-slate-500">Sắp xếp theo</span>
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value)}
              className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-slate-700 shadow-xs focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-100"
            >
              {Object.entries(SORTS).map(([k, [l]]) => (
                <option key={k} value={k}>
                  {l}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-xs font-semibold text-slate-500">
              Hiển thị: <b className="text-slate-800">{Math.min(limit, n)}</b> / {n} video
            </span>
            <input
              type="range"
              min={5}
              max={Math.max(5, n)}
              value={limit}
              onChange={(e) => setLimit(+e.target.value)}
              className="accent-blue-600"
            />
          </label>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full min-w-[650px] text-left text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-xs font-semibold uppercase tracking-wider text-slate-500">
              <tr>
                <th className="py-2.5 px-3">Thời gian</th>
                <th className="px-3">Tên tệp</th>
                <th className="px-3">Thời lượng</th>
                <th className="px-3">Độ phân giải</th>
                <th className="px-3">FPS</th>
                <th className="px-3">Frames</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {shown.map((v) => (
                <tr key={v.id} className="transition-colors hover:bg-slate-50">
                  <td className="py-2.5 px-3 text-slate-500 font-mono text-xs">{fmtDateTime(v.time)}</td>
                  <td className="px-3 font-semibold text-slate-800">{v.file}</td>
                  <td className="px-3 text-slate-600 font-medium">{v.duration}s</td>
                  <td className="px-3">
                    <span
                      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${
                        RESOLUTION_BADGE[v.resolution] || 'bg-slate-100 text-slate-700 border-slate-200'
                      }`}
                    >
                      <span
                        className="size-1.5 rounded-full"
                        style={{ backgroundColor: RESOLUTION_COLOR[v.resolution] || '#6366F1' }}
                      />
                      {v.resolution}
                    </span>
                  </td>
                  <td className="px-3 font-semibold text-emerald-600">{v.fps}</td>
                  <td className="px-3 text-slate-600 font-mono text-xs">{v.frames.toLocaleString('vi-VN')}</td>
                </tr>
              ))}
              {!n && (
                <tr>
                  <td colSpan={6} className="py-12 text-center text-slate-400">
                    Chưa có lịch sử nào. Hãy phân tích video để ghi nhận dữ liệu đầu tiên.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Aggregate Statistics Footer with Distinct Palette Colors */}
      <Card title="Chỉ số tổng hợp toàn thời gian" className="mt-6">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          {STAT_THEMES.map((item, idx) => {
            const I = item.icon;
            const val = statValues[idx];
            return (
              <div
                key={item.label}
                className="rounded-xl border border-slate-200 bg-white p-3.5 shadow-xs transition-transform hover:-translate-y-0.5"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-500">{item.label}</span>
                  <span
                    className="grid size-7 place-items-center rounded-md"
                    style={{ backgroundColor: `${item.color}18`, color: item.color }}
                  >
                    <I size={14} />
                  </span>
                </div>
                <div className="mt-2 text-xl font-bold text-slate-900">{val}</div>
              </div>
            );
          })}
        </div>
        <div className="mt-5 flex flex-wrap gap-3">
          <button
            onClick={exportCsv}
            disabled={!n}
            className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-xs transition-colors hover:bg-blue-700 disabled:opacity-40"
          >
            <FileDown size={16} /> Xuất tệp báo cáo CSV
          </button>
          <button
            onClick={onClear}
            disabled={!n}
            className="flex items-center gap-2 rounded-lg border border-red-200 bg-white px-4 py-2 text-sm font-semibold text-red-600 shadow-xs transition-colors hover:bg-red-50 disabled:opacity-40"
          >
            <Trash2 size={16} /> Xóa toàn bộ lịch sử
          </button>
        </div>
      </Card>
    </>
  );
}
