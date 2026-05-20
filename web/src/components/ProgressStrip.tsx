import { useEffect, useRef } from 'react'
import type { WindowRow } from '../types'

interface Props {
  windows: WindowRow[]
  currentIdx: number
  onSelect: (idx: number) => void
}

function blockColor(w: WindowRow, isCurrent: boolean): string {
  const base =
    w.human_label === 'chewing' ? 'bg-chewing' :
    w.human_label === 'rest'    ? 'bg-rest'    :
    w.human_label === 'bad_face' ? 'bg-bad'    :
    'bg-unlabeled'
  return base + (isCurrent ? ' ring-2 ring-white' : '')
}

export default function ProgressStrip({ windows, currentIdx, onSelect }: Props) {
  const labeled = windows.filter(w => w.human_label).length
  const currentRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    currentRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' })
  }, [currentIdx])

  return (
    <div className="flex flex-col gap-1 px-2 py-2" style={{ background: 'var(--surface)' }}>
      <div className="flex gap-px overflow-x-auto">
        {windows.map((w, i) => (
          <button
            key={w.id}
            ref={i === currentIdx ? currentRef : undefined}
            title={`Window ${i + 1}: ${w.human_label ?? 'unlabeled'}`}
            className={`flex-shrink-0 h-8 cursor-pointer rounded-sm transition-all ${blockColor(w, i === currentIdx)}`}
            style={{ width: `${Math.max(10, Math.floor(900 / windows.length))}px` }}
            onClick={() => onSelect(i)}
          />
        ))}
      </div>
      <div className="flex gap-4 text-xs" style={{ color: 'var(--muted)' }}>
        <span>{labeled} / {windows.length} labeled</span>
        <span className="label-chewing">■ chewing</span>
        <span className="label-rest">■ rest</span>
        <span className="label-bad">■ bad_face</span>
      </div>
    </div>
  )
}
