import { useEffect, useState } from 'react';
import { Download, Play, Trash2, X } from 'lucide-react';
import { Bar, Card, DropZone, PageHead } from '../ui';
import { mb } from '../data';

const BADGE = {
  Done: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  'Detecting...': 'bg-blue-50 text-blue-700 border-blue-200',
  'Uploading...': 'bg-blue-50 text-blue-700 border-blue-200',
  'Waiting...': 'bg-slate-100 text-slate-600 border-slate-200',
  Ready: 'bg-slate-100 text-slate-600 border-slate-200',
};

export default function BatchProcessing() {
  const [files, setFiles] = useState([]); // [{ id, file, progress }]
  const [running, setRunning] = useState(false);

  const status = (f) =>
    f.progress >= 100
      ? 'Done'
      : f.progress >= 20
      ? 'Detecting...'
      : f.progress > 0
      ? 'Uploading...'
      : running
      ? 'Waiting...'
      : 'Ready';

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setFiles((fs) => {
        const i = fs.findIndex((f) => f.progress < 100);
        return fs.map((f, j) =>
          j === i
            ? { ...f, progress: Math.min(100, f.progress + 8 + Math.floor(Math.random() * 10)) }
            : f
        );
      });
    }, 400);
    return () => clearInterval(id);
  }, [running]);

  useEffect(() => {
    if (running && files.every((f) => f.progress >= 100)) setRunning(false);
  }, [files, running]);

  const add = (list) =>
    setFiles((fs) => [...fs, ...list.map((file) => ({ id: crypto.randomUUID(), file, progress: 0 }))]);

  const total = files.reduce((s, f) => s + f.file.size, 0);
  const doneCount = files.filter((f) => f.progress >= 100).length;
  const overall = files.length
    ? Math.round(files.reduce((s, f) => s + f.progress, 0) / files.length)
    : 0;

  return (
    <>
      <PageHead
        title="Batch Processing"
        sub="Xử lý hàng loạt nhiều video giám sát cùng lúc từ hệ thống camera tự động."
      />
      <div className="grid gap-6 lg:grid-cols-5">
        {/* Upload and Queue Column (2/5) */}
        <div className="space-y-6 lg:col-span-2">
          <Card title="Tải lên danh sách video">
            <DropZone
              multiple
              onFiles={add}
              hint="Hỗ trợ chọn cùng lúc nhiều tệp: MP4, AVI, MOV, MKV | Tối đa 500MB/tệp"
            />
          </Card>

          <Card
            title={`Hàng đợi xử lý (${files.length})`}
            right={
              <button
                onClick={() => setFiles([])}
                disabled={running}
                className="flex items-center gap-1 text-xs font-semibold text-slate-500 hover:text-red-600 disabled:opacity-40 transition-colors"
              >
                <Trash2 size={14} /> Xóa danh sách
              </button>
            }
          >
            {files.length === 0 ? (
              <p className="py-8 text-center text-sm text-slate-400">
                Chưa có tệp nào. Kéo thả các video vào khung bên trên để tạo hàng đợi.
              </p>
            ) : (
              <ul className="max-h-64 divide-y divide-slate-100 overflow-auto text-sm">
                {files.map((f, i) => (
                  <li key={f.id} className="flex items-center gap-3 py-2.5">
                    <span className="w-5 font-mono text-xs text-slate-400">{i + 1}</span>
                    <span className="flex-1 truncate font-medium text-slate-800">{f.file.name}</span>
                    <span className="text-xs text-slate-500">{mb(f.file.size)} MB</span>
                    <button
                      disabled={running}
                      onClick={() => setFiles((fs) => fs.filter((x) => x.id !== f.id))}
                      aria-label="Remove"
                      className="text-slate-400 hover:text-red-500 transition-colors disabled:opacity-40"
                    >
                      <X size={15} />
                    </button>
                  </li>
                ))}
              </ul>
            )}

            <div className="mt-4 grid grid-cols-3 gap-3 text-center text-sm">
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div className="text-xs font-medium text-slate-500">Tổng số tệp</div>
                <div className="mt-1 text-lg font-bold text-slate-900">{files.length}</div>
              </div>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div className="text-xs font-medium text-slate-500">Dung lượng</div>
                <div className="mt-1 text-lg font-bold text-slate-900">{mb(total)} MB</div>
              </div>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div className="text-xs font-medium text-slate-500">Thời gian ước tính</div>
                <div className="mt-1 text-lg font-bold text-slate-900">
                  {Math.ceil((files.length * 30) / 60)} phút
                </div>
              </div>
            </div>

            <button
              disabled={!files.length || running}
              onClick={() => {
                setFiles((fs) => fs.map((f) => ({ ...f, progress: 0 })));
                setRunning(true);
              }}
              className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 py-3 font-semibold text-white shadow-xs transition-colors hover:bg-blue-700 disabled:opacity-40"
            >
              <Play size={18} /> Bắt đầu xử lý hàng loạt
            </button>
          </Card>
        </div>

        {/* Progress & Live Results Column (3/5) */}
        <div className="space-y-6 lg:col-span-3">
          <Card
            title="Tiến trình hàng đợi"
            right={
              <span className="rounded-full border border-blue-200 bg-blue-50 px-2.5 py-0.5 text-xs font-semibold text-blue-700">
                {running ? `Đang xử lý (${doneCount}/${files.length})` : 'Chờ lệnh'}
              </span>
            }
          >
            <div className="mb-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="mb-2 flex justify-between text-sm font-medium">
                <span className="text-slate-600">Tổng tiến độ</span>
                <span className="font-semibold text-blue-600">{overall}%</span>
              </div>
              <Bar value={overall} />
            </div>

            <div className="max-h-96 divide-y divide-slate-100 overflow-auto">
              {files.map((f) => {
                const st = status(f);
                return (
                  <div key={f.id} className="space-y-2 py-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="truncate font-medium text-slate-800">{f.file.name}</span>
                      <span
                        className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold ${
                          BADGE[st] || BADGE.Ready
                        }`}
                      >
                        {st}
                      </span>
                    </div>
                    <Bar value={f.progress} />
                  </div>
                );
              })}
            </div>
          </Card>
        </div>
      </div>
    </>
  );
}
