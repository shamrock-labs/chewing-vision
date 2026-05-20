import { useEffect, useRef } from 'react'
import { Chart, LineElement, PointElement, LinearScale, TimeScale, Tooltip, Legend, LineController, CategoryScale } from 'chart.js'
import zoomPlugin from 'chartjs-plugin-zoom'
import type { SignalPoint, WindowRow } from '../types'

Chart.register(LineElement, PointElement, LinearScale, TimeScale, Tooltip, Legend, LineController, CategoryScale, zoomPlugin)

interface Props {
  signals: SignalPoint[]
  windows: WindowRow[]
  currentIdx: number
  onWindowClick: (idx: number) => void
}

function normalize(arr: number[]): number[] {
  const finite = arr.filter(Number.isFinite)
  if (finite.length === 0) return arr.map(() => 0)
  const lo = Math.min(...finite), hi = Math.max(...finite)
  if (hi === lo) return arr.map(() => 0)
  return arr.map(v => Number.isFinite(v) ? (v - lo) / (hi - lo) : 0)
}

function downsample(pts: SignalPoint[], n = 3): SignalPoint[] {
  return pts.filter((_, i) => i % n === 0)
}

function bandColor(label: string | null | undefined): string {
  if (label === 'chewing') return 'rgba(74,158,237,0.13)'
  if (label === 'rest')    return 'rgba(34,197,94,0.10)'
  return 'rgba(96,96,128,0.06)'
}

export default function SignalChart({ signals, windows, currentIdx, onWindowClick }: Props) {
  const canvasRef    = useRef<HTMLCanvasElement>(null)
  const chartRef     = useRef<Chart | null>(null)
  const idxRef       = useRef(currentIdx)
  const windowsRef   = useRef(windows)

  // Keep refs in sync without recreating chart
  useEffect(() => { idxRef.current = currentIdx }, [currentIdx])
  useEffect(() => { windowsRef.current = windows }, [windows])

  // Redraw highlight only when currentIdx changes (no chart recreation)
  useEffect(() => {
    idxRef.current = currentIdx
    if (canvasRef.current?.isConnected) chartRef.current?.update('none')
  }, [currentIdx])

  // Create chart only when signals change
  useEffect(() => {
    if (!canvasRef.current || signals.length === 0) return
    if (chartRef.current) chartRef.current.destroy()

    const pts     = downsample(signals)
    const labels  = pts.map(p => p.t)
    const jawNorm = normalize(pts.map(p => p.jaw_open))
    const marNorm = normalize(pts.map(p => p.mar))

    const windowBandsPlugin = {
      id: 'windowBands',
      afterDraw(chart: Chart) {
        const { ctx, scales } = chart as any
        const xScale = scales['x'], yScale = scales['y']
        if (!xScale || !yScale) return
        for (const w of windowsRef.current) {
          const x0 = xScale.getPixelForValue(w.t_start)
          const x1 = xScale.getPixelForValue(w.t_end)
          const top = yScale.top, h = yScale.bottom - yScale.top
          ctx.save()
          ctx.fillStyle = bandColor(w.jaw_open_label)
          ctx.fillRect(x0, top, x1 - x0, h)
          if (w.jaw_open_label !== w.composite_label) {
            ctx.fillStyle = 'rgba(239,68,68,0.55)'
            ctx.fillRect(x0, top, x1 - x0, 3)
          }
          ctx.restore()
        }
      },
    }

    const highlightPlugin = {
      id: 'windowHighlight',
      afterDraw(chart: Chart) {
        const win = windowsRef.current[idxRef.current]
        if (!win) return
        const { ctx, scales } = chart as any
        const xScale = scales['x'], yScale = scales['y']
        if (!xScale || !yScale) return
        const x0 = xScale.getPixelForValue(win.t_start)
        const x1 = xScale.getPixelForValue(win.t_end)
        ctx.save()
        ctx.fillStyle   = 'rgba(245,158,11,0.15)'
        ctx.strokeStyle = 'rgba(245,158,11,0.7)'
        ctx.lineWidth   = 1
        ctx.fillRect  (x0, yScale.top, x1 - x0, yScale.bottom - yScale.top)
        ctx.strokeRect(x0, yScale.top, x1 - x0, yScale.bottom - yScale.top)
        ctx.restore()
      },
    }

    chartRef.current = new Chart(canvasRef.current, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'jaw_open', data: jawNorm, borderColor: '#4a9eed', borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
          { label: 'MAR',      data: marNorm, borderColor: '#22c55e', borderWidth: 1,   pointRadius: 0, tension: 0.2 },
        ],
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        onClick(_evt, _elems, chart) {
          const xScale = (chart as any).scales['x']
          if (!xScale) return
          const mouseX = (_evt as any).native?.offsetX ?? 0
          const t = xScale.getValueForPixel(mouseX)
          const i = windowsRef.current.findIndex(w => t >= w.t_start && t <= w.t_end)
          if (i >= 0) onWindowClick(i)
        },
        scales: {
          x: { type: 'linear', ticks: { color: '#606080', maxTicksLimit: 8 }, grid: { color: '#1e1e30' } },
          y: { min: 0, max: 1, ticks: { color: '#606080' }, grid: { color: '#1e1e30' } },
        },
        plugins: {
          legend: { labels: { color: '#a0a0c0', font: { size: 11 } } },
          tooltip: { enabled: false },
          zoom: {
            zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
            pan:  { enabled: true, mode: 'x' },
          },
        },
      },
      plugins: [windowBandsPlugin, highlightPlugin],
    })

    const el = canvasRef.current
    const reset = () => chartRef.current?.resetZoom()
    el.addEventListener('dblclick', reset)
    return () => {
      el.removeEventListener('dblclick', reset)
      chartRef.current?.destroy()
      chartRef.current = null
    }
  }, [signals])   // eslint-disable-line react-hooks/exhaustive-deps

  return <canvas ref={canvasRef} style={{ width: '100%', height: '100%', cursor: 'crosshair' }} />
}
