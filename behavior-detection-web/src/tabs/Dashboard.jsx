import { CalendarClock, Gauge, Timer, Video } from 'lucide-react';
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import { Card, PageHead, Stat } from '../ui';
import { fmtTime, RESOLUTION_BADGE, RESOLUTION_COLOR } from '../data';

const AXIS_STYLE = { stroke: '#94a3b8', fontSize: 12 };
const TOOLTIP_STYLE = {
  backgroundColor: '#FFFFFF',
  border: '1px solid #E2E8F0',
  borderRadius: '8px',
  boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
  color: '#0F172A',
  fontSize: '13px',
};

export default function Dashboard({ videos }) {
  const n = videos.length;
  const totalDur = videos.reduce((s, v) => s + v.duration, 0);
  const avgFps = n ? videos.reduce((s, v) => s + v.fps, 0) / n : 0;
  const last = videos.reduce((m, v) => Math.max(m, v.time), 0);
  const recent = [...videos].sort((a, b) => b.time - a.time).slice(0, 10);

  const counts = {};
  videos.forEach((v) => {
    counts[v.resolution] = (counts[v.resolution] || 0) + 1;
  });
  const pie = Object.entries(counts).map(([name, value]) => ({ name, value }));
  const points = videos.map((v) => ({
    t: v.time,
    d: v.duration,
    file: v.file,
    resolution: v.resolution,
  }));

  return (
    <>
      <PageHead
        title="Dashboard"
        sub="Tổng quan hệ thống, thống kê hiệu suất vận hành và phân bố độ phân giải camera."
      />

      {/* Top 4 Metrics - Mỗi thẻ một màu chủ đạo riêng biệt */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="Số Video Đã Phân Tích" value={n} icon={Video} color="#6366F1" />
        <Stat label="Tổng Thời Lượng" value={`${totalDur.toFixed(1)}s`} icon={Timer} color="#10B981" />
        <Stat label="Tốc Độ Xử Lý TB" value={`${avgFps.toFixed(1)} FPS`} icon={Gauge} color="#F59E0B" />
        <Stat label="Lần Phân Tích Gần Nhất" value={last ? fmtTime(last) : '--:--'} icon={CalendarClock} color="#0284C7" />
      </div>

      {/* Charts Grid */}
      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        {/* Timeline Scatter Plot with Color-Coded Dots */}
        <Card
          title="Dòng thời gian xử lý (Timeline)"
          right={
            <div className="flex items-center gap-3 text-xs font-semibold text-slate-500">
              <span className="flex items-center gap-1.5">
                <span className="size-2 rounded-full" style={{ backgroundColor: RESOLUTION_COLOR['1920x1080'] }} /> 1080p
              </span>
              <span className="flex items-center gap-1.5">
                <span className="size-2 rounded-full" style={{ backgroundColor: RESOLUTION_COLOR['1280x720'] }} /> 720p
              </span>
              <span className="flex items-center gap-1.5">
                <span className="size-2 rounded-full" style={{ backgroundColor: RESOLUTION_COLOR['854x480'] }} /> 480p
              </span>
            </div>
          }
        >
          <ResponsiveContainer width="100%" height={280}>
            <ScatterChart margin={{ left: -10, right: 16, top: 8 }}>
              <XAxis
                dataKey="t"
                type="number"
                domain={['auto', 'auto']}
                tickFormatter={fmtTime}
                name="Thời gian"
                {...AXIS_STYLE}
              />
              <YAxis dataKey="d" type="number" name="Thời lượng" unit="s" {...AXIS_STYLE} />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                content={({ payload }) =>
                  payload?.[0] && (
                    <div style={TOOLTIP_STYLE} className="p-2.5">
                      <div className="font-semibold text-slate-800">{payload[0].payload.file}</div>
                      <div className="mt-1 text-xs text-slate-500">
                        {fmtTime(payload[0].payload.t)} · Thời lượng: {payload[0].payload.d}s
                      </div>
                      <div className="mt-1 text-xs font-semibold" style={{ color: RESOLUTION_COLOR[payload[0].payload.resolution] || '#6366F1' }}>
                        Độ phân giải: {payload[0].payload.resolution}
                      </div>
                    </div>
                  )
                }
              />
              <Scatter data={points}>
                {points.map((entry, index) => (
                  <Cell
                    key={`dot-${index}`}
                    fill={RESOLUTION_COLOR[entry.resolution] || '#6366F1'}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </Card>

        {/* Resolution Donut Chart with Distinct Colors */}
        <Card title="Phân bố độ phân giải camera">
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={pie}
                dataKey="value"
                nameKey="name"
                outerRadius={95}
                innerRadius={50}
                label={({ percent }) => `${(percent * 100).toFixed(0)}%`}
              >
                {pie.map((entry, i) => (
                  <Cell
                    key={`slice-${i}`}
                    fill={RESOLUTION_COLOR[entry.name] || '#6366F1'}
                    stroke="#FFFFFF"
                    strokeWidth={2}
                  />
                ))}
              </Pie>
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Recent Videos Table */}
      <Card title="Danh sách video xử lý gần đây" className="mt-6">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[500px] text-left text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-xs font-semibold uppercase tracking-wider text-slate-500">
              <tr>
                <th className="py-2.5 px-3">Thời gian</th>
                <th className="px-3">Tên tệp</th>
                <th className="px-3">Thời lượng</th>
                <th className="px-3">Độ phân giải</th>
                <th className="px-3">Tốc độ (FPS)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {recent.map((v) => (
                <tr key={v.id} className="transition-colors hover:bg-slate-50">
                  <td className="py-2.5 px-3 text-slate-500 font-mono text-xs">{fmtTime(v.time)}</td>
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
                  <td className="px-3 font-semibold text-emerald-600">{v.fps} FPS</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
