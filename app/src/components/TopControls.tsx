import type { UserTrajectory } from "../data/trajectories";

interface Props {
  availableUsers: UserTrajectory[];
  selectedUser: UserTrajectory | null;
  sampleUserIds: number[];
  onRandomize: () => void;
  onSelectUser: (userId: number) => void;
  onTogglePlay: () => void;
  onRevealChange: (value: number) => void;
  onSpeedChange: (speed: number) => void;
  playing: boolean;
  revealCount: number;
  maxSteps: number;
  speed: number;
  totalCheckins: number;
  totalDistanceKm: number;
}

export default function TopControls({
  availableUsers,
  selectedUser,
  sampleUserIds,
  onRandomize,
  onSelectUser,
  onTogglePlay,
  onRevealChange,
  onSpeedChange,
  playing,
  revealCount,
  maxSteps,
  speed,
  totalCheckins,
  totalDistanceKm,
}: Props) {
  return (
    <section className="rounded-[28px] border border-slate-200/80 bg-white/85 px-4 py-4 shadow-[0_18px_50px_rgba(15,23,42,0.06)] backdrop-blur sm:px-6">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
        <div className="max-w-4xl">
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-indigo-500">
            Next POI Recommendation Visual Lab
          </p>
          <h2 className="mt-1 text-2xl font-semibold tracking-tight text-slate-900">
            上图下表 · 单用户展示 · 推荐命中高亮
          </h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-500">
            每次只展示一位样本用户，顶部控制条负责播放回放，底部卡片区负责画像、类别分布和推荐榜单。
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={onRandomize}
            className="inline-flex items-center gap-2 rounded-2xl bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-slate-800 active:scale-95"
          >
            随机抽 1 位用户
          </button>
          <div className="rounded-2xl bg-slate-100 px-4 py-2.5 text-xs text-slate-500">
            样本池 {sampleUserIds.length} 人
          </div>
          <div className="rounded-2xl bg-slate-100 px-4 py-2.5 text-xs text-slate-500">
            当前 1 人
          </div>
        </div>
      </div>

      <div className="mt-4 grid gap-3 lg:grid-cols-[1.3fr_auto_1fr_auto] lg:items-center">
        <label className="grid gap-1.5 text-xs font-medium text-slate-500">
          用户
          <select
            value={selectedUser?.userId ?? selectedUser?.userIdx ?? ""}
            onChange={(event) => onSelectUser(Number(event.target.value))}
            className="rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 outline-none transition focus:border-indigo-400"
          >
            {availableUsers.map((user) => (
              <option key={user.userId ?? user.userIdx} value={user.userId ?? user.userIdx}>
                {user.userLabel}
              </option>
            ))}
          </select>
        </label>

        <button
          onClick={onTogglePlay}
          className="inline-flex items-center justify-center gap-2 rounded-2xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-slate-50 active:scale-95"
        >
          {playing ? "暂停回放" : revealCount >= maxSteps ? "重新播放" : "播放轨迹"}
        </button>

        <div className="rounded-2xl border border-slate-200 bg-white px-4 py-2.5">
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={1}
              max={maxSteps}
              value={revealCount}
              onChange={(event) => onRevealChange(Number(event.target.value))}
              className="h-1.5 flex-1 cursor-pointer accent-indigo-600"
            />
            <span className="w-16 text-right text-xs tabular-nums text-slate-500">
              {revealCount}/{maxSteps}
            </span>
          </div>
        </div>

        <select
          value={speed}
          onChange={(event) => onSpeedChange(Number(event.target.value))}
          className="rounded-2xl border border-slate-200 bg-white px-4 py-2.5 text-sm text-slate-700 outline-none transition focus:border-indigo-400"
        >
          <option value={2000}>慢</option>
          <option value={1200}>中</option>
          <option value={600}>快</option>
        </select>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatChip label="选中用户" value={selectedUser ? "1 人" : "0 人"} />
        <StatChip label="签到点" value={`${totalCheckins} 次`} />
        <StatChip label="轨迹里程" value={`${totalDistanceKm.toFixed(1)} km`} />
        <StatChip label="样本池" value={`${availableUsers.length} 人`} />
      </div>
    </section>
  );
}

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-slate-50 px-4 py-3">
      <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400">{label}</p>
      <p className="mt-1 text-lg font-semibold text-slate-900">{value}</p>
    </div>
  );
}