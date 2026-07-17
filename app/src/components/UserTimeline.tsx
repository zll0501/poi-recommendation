import type { UserTrajectory } from "../data/trajectories";
import { totalDistance } from "../data/trajectories";

interface Props {
  user: UserTrajectory;
  revealCount: number;
  onSelectStop: (lat: number, lon: number) => void;
}

export default function UserTimeline({ user, revealCount, onSelectStop }: Props) {
  const dist = totalDistance(user.checkins);

  return (
    <div className="rounded-2xl border border-slate-200 bg-white/80 p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className="inline-block h-3 w-3 rounded-full ring-2 ring-white"
            style={{ background: user.color, boxShadow: `0 0 0 2px ${user.color}33` }}
          />
          <div>
            <p className="text-sm font-semibold text-slate-800">{user.userLabel}</p>
            <p className="text-xs text-slate-400">{user.borough}</p>
          </div>
        </div>
        <div className="text-right text-xs text-slate-400">
          <p>{user.weekday}</p>
          <p>{user.checkins.length} 次签到</p>
          <p>{dist.toFixed(1)} km</p>
        </div>
      </div>

      <ol className="space-y-2">
        {user.checkins.map((c, idx) => {
          const active = idx < revealCount;
          return (
            <li key={c.order}>
              <button
                onClick={() => onSelectStop(c.lat, c.lon)}
                className={`flex w-full items-center gap-3 rounded-xl border px-2.5 py-1.5 text-left transition ${
                  active
                    ? "border-slate-200 bg-slate-50 hover:bg-slate-100"
                    : "border-transparent opacity-40"
                }`}
              >
                <span
                  className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-bold text-white"
                  style={{ background: user.color }}
                >
                  {c.order}
                </span>
                <span className="text-base leading-none">{c.emoji}</span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-xs font-medium text-slate-700">
                    {c.poiName}
                    <span className="ml-1 text-slate-400">· {c.category}</span>
                  </span>
                </span>
                <span className="shrink-0 text-[11px] tabular-nums text-slate-400">{c.timeLabel}</span>
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
