import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Activity, ArrowLeft, Footprints, HeartPulse, PieChart as PieIcon, ShieldAlert, Swords } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Card, Stat } from '../ui';
import { BEHAVIOR_COLOR as C, fmtDateTime, fmtTime } from '../data';

const TYPES = ['fighting', 'falling', 'loitering'];

const BADGE_STYLES = {
  fighting: 'bg-red-50 text-red-700 border-red-200',
  falling: 'bg-amber-50 text-amber-700 border-amber-200',
  loitering: 'bg-blue-50 text-blue-700 border-blue-200',
  normal: 'bg-emerald-50 text-emerald-700 border-emerald-200',
};

const AXIS_STYLE = { stroke: '#94a3b8', fontSize: 12 };
const TOOLTIP_STYLE = {
  backgroundColor: '#FFFFFF',
  border: '1px solid #E2E8F0',
  borderRadius: '8px',
  boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
  color: '#0F172A',
  fontSize: '13px',
};

let uid = 0;
const mk = (ts) => ({
  id: ++uid,
  Timestamp: ts,
  Track_ID: 1 + Math.floor(Math.random() * 40),
  Behavior: TYPES[Math.floor(Math.random() * 3)],
  Confidence: +(0.6 + Math.random() * 0.39).toFixed(2),
});
const seed = () =>
  Array.from({ length: 30 }, () => Date.now() - Math.random() * 15 * 60000)
    .sort((a, b) => a - b)
    .map((t) => mk(t));

export default function Analysis() {
  const [events, setEvents] = useState(seed);

  useEffect(() => {
    const fetchEvents = () => {
      fetch('/api/events')
        .then((r) => {
          if (!r.ok) throw new Error('API returned ' + r.status);
          return r.json();
        })
        .then((data) => {
          if (Array.isArray(data) && data.length > 0) {
            setEvents(data);
          }
        })
        .catch(() => {
          setEvents((evs) =>
            Math.random() < 0.6
              ? [...evs, ...Array.from({ length: 1 + Math.floor(Math.random() * 2) }, () => mk(Date.now()))]
              : evs
          );
        });
    };

    fetchEvents();
    const id = setInterval(fetchEvents, 3000);
    return () => clearInterval(id);
  }, []);

  const count = (b) => events.filter((e) => e.Behavior === b).length;
  const fresh = events.some((e) => Date.now() - e.Timestamp < 60000);
  const byMin = {};
  events.forEach((e) => {
    const k = fmtTime(e.Timestamp);
    (byMin[k] ??= { minute: k, fighting: 0, falling: 0, loitering: 0 })[e.Behavior]++;
  });
  const bars = Object.values(byMin).slice(-15);
  const pie = TYPES.map((t) => ({ name: t, value: count(t) }));
  const log = events.slice(-50).reverse();

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      <div className="mx-auto max-w-7xl space-y-6 p-5 md:p-8">
        {/* Header */}
        <header className="flex items-center gap-3">
          <Link
            to="/"
            className="grid size-9 place-items-center rounded-lg border border-slate-200 bg-white text-slate-600 shadow-xs transition-colors hover:border-slate-300 hover:text-slate-900"
            aria-label="Back"
          >
            <ArrowLeft size={18} />
          </Link>
          <div className="flex-1">
            <h1 className="text-2xl font-bold tracking-tight text-slate-900">CCTV Live Control Room</h1>
            <p className="text-sm text-slate-500">Giám sát & phát hiện hành vi bất thường theo thời gian thực.</p>
          </div>
          <span className="flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700">
            <span className="size-2 animate-pulse rounded-full bg-emerald-500" /> Auto-refresh 3s
          </span>
        </header>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <Stat label="Tổng Sự Cố Nay" value={events.length} icon={ShieldAlert} alert={fresh} />
          <Stat label="Bạo lực (Fighting)" value={count('fighting')} icon={Swords} color={C.fighting} />
          <Stat label="Té ngã (Falling)" value={count('falling')} icon={HeartPulse} color={C.falling} />
          <Stat label="Lảng vảng (Loitering)" value={count('loitering')} icon={Footprints} color={C.loitering} />
        </div>

        {/* Charts Grid */}
        <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
          <Card title="Dòng thời gian sự cố (Timeline)" icon={Activity}>
            <ResponsiveContainer width="100%" height={290}>
              <BarChart data={bars}>
                <CartesianGrid stroke="#F1F5F9" vertical={false} />
                <XAxis dataKey="minute" {...AXIS_STYLE} />
                <YAxis allowDecimals={false} {...AXIS_STYLE} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Legend />
                {TYPES.map((t) => (
                  <Bar key={t} dataKey={t} stackId="a" fill={C[t]} isAnimationActive={false} radius={[2, 2, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Tỷ trọng hành vi bất thường" icon={PieIcon}>
            <ResponsiveContainer width="100%" height={290}>
              <PieChart>
                <Pie data={pie} dataKey="value" nameKey="name" innerRadius="45%" outerRadius="80%" isAnimationActive={false}>
                  {pie.map((p) => (
                    <Cell key={p.name} fill={C[p.name]} stroke="#FFFFFF" strokeWidth={2} />
                  ))}
                </Pie>
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>

        {/* Event Logs Table */}
        <Card title="Nhật ký sự kiện gần đây (50 cảnh báo mới nhất)">
          <div className="max-h-96 overflow-auto">
            <table className="w-full text-left text-sm">
              <thead className="sticky top-0 bg-slate-50 border-b border-slate-200 text-xs font-semibold uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="py-2.5 px-3">Thời gian</th>
                  <th className="px-3">Track ID</th>
                  <th className="px-3">Hành vi</th>
                  <th className="px-3">Độ tin cậy</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {log.map((e) => {
                  const badgeClass = BADGE_STYLES[e.Behavior] || BADGE_STYLES.normal;
                  return (
                    <tr key={e.id} className="transition-colors hover:bg-slate-50">
                      <td className="py-2.5 px-3 font-medium text-slate-700">{fmtDateTime(e.Timestamp)}</td>
                      <td className="px-3 text-slate-600 font-mono text-xs">#{e.Track_ID}</td>
                      <td className="px-3">
                        <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold capitalize ${badgeClass}`}>
                          <span className="size-1.5 rounded-full" style={{ backgroundColor: C[e.Behavior] || '#10B981' }} />
                          {e.Behavior}
                        </span>
                      </td>
                      <td className="px-3 text-slate-600 font-medium">
                        {(Number(e.Confidence) * 100).toFixed(0)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}
