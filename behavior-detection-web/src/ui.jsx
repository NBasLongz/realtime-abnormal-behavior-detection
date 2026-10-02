import { useRef } from 'react';
import { FolderOpen, UploadCloud } from 'lucide-react';

export const PageHead = ({ title, sub }) => (
  <div className="mb-6">
    <h1 className="text-2xl font-bold tracking-tight text-slate-900">{title}</h1>
    <p className="mt-1 text-sm text-slate-500">{sub}</p>
  </div>
);

export const Card = ({ title, icon: Icon, right, className = '', children }) => (
  <section className={`rounded-xl border border-slate-200 bg-white p-5 shadow-xs transition-shadow hover:shadow-sm ${className}`}>
    {(title || right) && (
      <header className="mb-4 flex items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <h3 className="flex items-center gap-2 text-base font-semibold text-slate-900">
          {Icon && <Icon size={18} className="text-blue-600" />}
          {title}
        </h3>
        {right}
      </header>
    )}
    {children}
  </section>
);

// Thẻ chỉ số linh hoạt màu sắc theo từng thành phần (Category-specific colors)
export const Stat = ({ label, value, color = '#2563EB', icon: Icon, alert, className = '' }) => (
  <div
    className={`rounded-xl border p-4.5 transition-all shadow-xs hover:shadow-sm ${
      alert
        ? 'border-red-200 bg-red-50/70'
        : 'border-slate-200/90 bg-white hover:border-slate-300'
    } ${className}`}
  >
    <div className="flex items-center justify-between">
      <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
        {label}
      </span>
      {Icon && (
        <span
          className="grid size-9 place-items-center rounded-lg transition-transform"
          style={{
            backgroundColor: alert ? '#FEE2E2' : `${color}18`,
            color: alert ? '#DC2626' : color,
          }}
        >
          <Icon size={18} />
        </span>
      )}
    </div>
    <div
      className={`mt-2.5 text-2xl font-bold tracking-tight ${
        alert ? 'text-red-700' : 'text-slate-900'
      }`}
    >
      {value}
    </div>
  </div>
);

export const Bar = ({ value, className = 'bg-blue-600' }) => (
  <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
    <div
      className={`h-full rounded-full transition-all duration-300 ${className}`}
      style={{ width: `${value}%` }}
    />
  </div>
);

export function DropZone({ onFiles, multiple, hint }) {
  const ref = useRef(null);
  return (
    <div
      onClick={() => ref.current.click()}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        onFiles([...e.dataTransfer.files]);
      }}
      className="group cursor-pointer rounded-xl border-2 border-dashed border-slate-300 bg-slate-50/50 p-8 text-center transition-all hover:border-blue-500 hover:bg-blue-50/30"
    >
      <div className="mx-auto mb-3 grid size-12 place-items-center rounded-full bg-blue-50 text-blue-600 transition-transform group-hover:scale-110">
        <UploadCloud size={24} />
      </div>
      <p className="font-semibold text-slate-800">
        {multiple ? 'Kéo thả nhiều video vào đây' : 'Upload Video'}
      </p>
      <p className="mt-0.5 text-sm text-slate-500">hoặc nhấp để chọn từ máy tính</p>
      <p className="mt-2 text-xs text-slate-400">{hint}</p>
      <input
        ref={ref}
        type="file"
        multiple={multiple}
        accept=".mp4,.avi,.mov,.mkv,video/*"
        hidden
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => {
          onFiles([...e.target.files]);
          e.target.value = '';
        }}
      />
      <div className="mt-4">
        <span className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-xs transition-colors hover:bg-blue-700">
          <FolderOpen size={16} /> Chọn tệp{multiple ? ' (Nhiều video)' : ''}
        </span>
      </div>
    </div>
  );
}
