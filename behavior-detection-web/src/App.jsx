import { Route, Routes } from 'react-router-dom';
import AdminApp from './pages/AdminApp';
import Analysis from './pages/Analysis';

// Hai trang độc lập (tương đương web/app.py và web/pages/analysis.py)
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<AdminApp />} />
      <Route path="/analysis" element={<Analysis />} />
    </Routes>
  );
}
