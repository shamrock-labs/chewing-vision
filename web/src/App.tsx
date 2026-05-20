import { BrowserRouter, Routes, Route } from 'react-router-dom'
import SessionList from './pages/SessionList'
import Annotate from './pages/Annotate'
import Results from './pages/Results'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<SessionList />} />
        <Route path="/annotate/:sessionId" element={<Annotate />} />
        <Route path="/results" element={<Results />} />
      </Routes>
    </BrowserRouter>
  )
}
