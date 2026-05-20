import { useEffect, useRef } from 'react'
import { Chart, LineElement, PointElement, LinearScale, Tooltip, Legend, LineController, CategoryScale } from 'chart.js'
import zoomPlugin from 'chartjs-plugin-zoom'
import type { ImuPoint, WindowRow } from '../types'

Chart.register(LineElement, PointElement, LinearScale, Tooltip, Legend, LineController, CategoryScale, zoomPlugin)

interface Props {
  imu: ImuPoint[]
  windows: WindowRow[]
  currentIdx: number
  onWindowClick: (idx: number) => void
}

export default function ImuChart({ imu, windows, currentIdx, onWindowClick }: Props) {
  const canvasRef  = useRef<HTMLCanvasElement>(null)
  const chartRef   = useRef<Chart | null>(null)
  const idxRef     = useRef(currentIdx)
  const windowsRef = useRef(windows)

  useEffect(() => { windowsRef.current = windows }, [windows])

  // Highlight update only — no chart recreation
  useEffect(() => {
    idxRef.current = currentIdx
    if (canvasRef.current?.isConnected) chartRef.current?.update('none')
  }, [currentIdx])

  // Create chart only when imu data changes
  useEffect(() => {
    if (!canvasRef.current || imu.length === 0) return
    if (chartRef.current) chartRef.current.destroy()

    const labels = imu.map(p => p.t)

    const windowBandsPlugin = {
      id: 'imuWindowBands',
      afterDraw(chart: Chart) {
        const { ctx, scales } = chart as any
        const x = scales['x'], y = scales['y']
        if (!x || !y) return
        for (const w of windowsRef.current) {
          if (w.jaw_open_label !== w.composite_label) {
            const x0 = x.getPixelForValue(w.t_start)
            const x1 = x.getPixelForValue(w.t_end)
            ctx.save()
            ctx.fillStyle = 'rgba(239,68,68,0.07)'
            ctx.fillRect(x0, y.top, x1 - x0, y.bottom - y.top)
            ctx.fillStyle = 'rgba(239,68,68,0.5)'
            ctx.fillRect(x0, y.top, x1 - x0, 2)
            ctx.restore()
          }
        }
      },
    }

    const highlightPlugin = {
      id: 'imuHighlight',
      afterDraw(chart: Chart) {
        const win = windowsRef.current[idxRef.current]
        if (!win) return
        const { ctx, scales } = chart as any
        const x = scales['x'], y = scales['y']
        if (!x || !y) return
        const x0 = x.getPixelForValue(win.t_start)
        const x1 = x.getPixelForValue(win.t_end)
        ctx.save()
        ctx.fillStyle   = 'rgba(245,158,11,0.15)'
        ctx.strokeStyle = 'rgba(245,158,11,0.7)'
        ctx.lineWidth   = 1
        ctx.fillRect  (x0, y.top, x1 - x0, y.bottom - y.top)
        ctx.strokeRect(x0, y.top, x1 - x0, y.bottom - y.top)
        ctx.restore()
      },
    }

    chartRef.current = new Chart(canvasRef.current, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'rot_z', data: imu.map(p => p.rot_z), borderColor: '#f97316', borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
          { label: 'rot_x', data: imu.map(p => p.rot_x), borderColor: '#a78bfa', borderWidth: 1,   pointRadius: 0, tension: 0.2 },
          { label: 'rot_y', data: imu.map(p => p.rot_y), borderColor: '#34d399', borderWidth: 1,   pointRadius: 0, tension: 0.2 },
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
          x: { type: 'linear', ticks: { color: '#606080', maxTicksLimit: 6 }, grid: { color: '#1e1e30' } },
          y: { ticks: { color: '#606080', maxTicksLimit: 4 }, grid: { color: '#1e1e30' } },
        },
        plugins: {
          legend: { labels: { color: '#a0a0c0', font: { size: 10 } } },
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
  }, [imu])   // eslint-disable-line react-hooks/exhaustive-deps

  return <canvas ref={canvasRef} style={{ width: '100%', height: '100%', cursor: 'crosshair' }} />
}
