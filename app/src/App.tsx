import { useEffect, useMemo, useState } from "react";
import TrajectoryMap from "./components/TrajectoryMap";
import UserTimeline from "./components/UserTimeline";
import {
  loadTrajectoryViewData,
  pickRandomUsers,
  totalDistance,
  type UserTrajectory,
} from "./data/trajectories";

export default function App() {
  const [availableUsers, setAvailableUsers] = useState<UserTrajectory[]>([]);
  const [selected, setSelected] = useState<UserTrajectory[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [revealCount, setRevealCount] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1200);
  const [focusPoint, setFocusPoint] = useState<{ lat: number; lon: number } | null>(null);

  useEffect(() => {
    let mounted = true;

    loadTrajectoryViewData()
      .then((users) => {
        if (!mounted) return;
        setAvailableUsers(users);
        setSelected(pickRandomUsers(users, 2, 2));
        setLoading(false);
      })
      .catch((error: unknown) => {
        if (!mounted) return;
        setLoadError(error instanceof Error ? error.message : "Failed to load trajectory data");
        setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, []);

  const maxSteps = useMemo(
    () => Math.max(1, ...selected.map((u) => u.checkins.length)),
    [selected]
  );

  useEffect(() => {
    setRevealCount(maxSteps);
    setPlaying(false);
  }, [selected, maxSteps]);

  useEffect(() => {
    if (selected.length === 0) return;
    if (!playing) return;
    if (revealCount >= maxSteps) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setRevealCount((n) => Math.min(maxSteps, n + 1)), speed);
    return () => clearTimeout(t);
  }, [playing, revealCount, maxSteps, speed]);

  const reroll = () => {
    setSelected(pickRandomUsers(availableUsers, 1, 3));
    setFocusPoint(null);
  };

  const togglePlay = () => {
    if (revealCount >= maxSteps) setRevealCount(1);
    setPlaying((p) => !p);
  };

  const totalCheckins = selected.reduce((s, u) => s + u.checkins.length, 0);
  const totalKm = selected.reduce((s, u) => s + totalDistance(u.checkins), 0);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-50 via-white to-indigo-50/40 text-slate-600">
        正在加载真实轨迹数据…
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-50 via-white to-indigo-50/40 px-6 text-center text-slate-700">
        <div className="max-w-xl rounded-2xl border border-rose-200 bg-white/90 p-6 shadow-sm">
          <h1 className="text-xl font-bold text-rose-700">轨迹数据加载失败</h1>
          <p className="mt-2 text-sm text-slate-500">{loadError}</p>
          <p className="mt-4 text-sm text-slate-500">
            请先运行数据生成脚本，确保 app/public/data/trajectories_view.json 存在。
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-indigo-50/40 text-slate-900">
      {/* Header */}
      <header className="border-b border-slate-200/70 bg-white/70 backdrop-blur">
        <div className="mx-auto max-w-[1400px] px-6 py-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-indigo-500">
                Figure 3 · User Trajectory Visualization
              </p>
              <h1 className="mt-1 text-2xl font-bold tracking-tight text-slate-900">
                用户历史签到轨迹地图
              </h1>
              <p className="mt-1 max-w-2xl text-sm leading-relaxed text-slate-500">
                数据来源：<code className="rounded bg-slate-100 px-1 py-0.5 text-[12px]">train_encoded.csv</code>
                与 <code className="rounded bg-slate-100 px-1 py-0.5 text-[12px]">poi_metadata.csv</code>
                生成的展示样本（Foursquare NYC Check-in）。若把全部用户签到点连成线，会得到一张没有展示价值的
                <span className="font-medium text-slate-700">"蜘蛛网"</span>。
                这里采用<span className="font-medium text-slate-700">随机抽取 1~3 位用户</span>的方式，
                按时间顺序还原其真实签到路径 —— 用户行为呈现出清晰的
                <span className="font-medium text-indigo-600"> 时空连续性（spatio-temporal continuity）</span>，
                这正是 <span className="font-medium text-indigo-600">Next POI Recommendation</span> 建模的核心依据。
              </p>
            </div>

            <div className="flex shrink-0 flex-col items-end gap-2">
              <button
                onClick={reroll}
                className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-200 transition hover:bg-indigo-500 active:scale-95"
              >
                <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                  <path d="M4 4v5h5" strokeLinecap="round" strokeLinejoin="round" />
                  <path
                    d="M4.5 9A8 8 0 1 1 6 16.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                随机抽取 1~3 位用户
              </button>
              <p className="text-xs text-slate-400">
                当前样本池共 {availableUsers.length} 位代表性用户，每次随机展示其中一部分
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] px-6 py-6">
        {/* Stats bar */}
        <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatCard label="本次展示用户数" value={`${selected.length} 人`} accent="text-indigo-600" />
          <StatCard label="签到点总数" value={`${totalCheckins} 次`} accent="text-emerald-600" />
          <StatCard label="轨迹总里程" value={`${totalKm.toFixed(1)} km`} accent="text-amber-600" />
          <StatCard label="数据集全量样本" value="1,077 用户 / 22.7万 签到" accent="text-rose-600" />
        </div>

        <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_360px]">
          {/* Map */}
          <div className="flex flex-col gap-3">
            <div className="relative h-[560px] overflow-hidden rounded-2xl border border-slate-200 shadow-sm">
              <TrajectoryMap users={selected} revealCount={revealCount} focusPoint={focusPoint} />

              {/* legend */}
              <div className="pointer-events-none absolute left-3 top-3 z-[500] rounded-xl bg-white/90 px-3 py-2 text-xs shadow backdrop-blur">
                <p className="mb-1 font-semibold text-slate-600">图例</p>
                <div className="flex flex-col gap-1">
                  {selected.map((u) => (
                    <div key={u.userIdx} className="flex items-center gap-1.5">
                      <span className="h-2.5 w-2.5 rounded-full" style={{ background: u.color }} />
                      <span className="text-slate-600">{u.userLabel}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* playback controls */}
            <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-slate-200 bg-white/80 px-4 py-3 shadow-sm">
              <button
                onClick={togglePlay}
                className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 px-3.5 py-2 text-xs font-semibold text-white transition hover:bg-slate-700 active:scale-95"
              >
                {playing ? (
                  <>
                    <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="currentColor">
                      <rect x="6" y="5" width="4" height="14" />
                      <rect x="14" y="5" width="4" height="14" />
                    </svg>
                    暂停
                  </>
                ) : (
                  <>
                    <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M8 5v14l11-7z" />
                    </svg>
                    {revealCount >= maxSteps ? "重新播放轨迹" : "播放轨迹动画"}
                  </>
                )}
              </button>

              <input
                type="range"
                min={1}
                max={maxSteps}
                value={revealCount}
                onChange={(e) => {
                  setPlaying(false);
                  setRevealCount(Number(e.target.value));
                }}
                className="h-1.5 flex-1 min-w-[140px] cursor-pointer accent-indigo-600"
              />
              <span className="w-16 shrink-0 text-right text-xs tabular-nums text-slate-500">
                {revealCount}/{maxSteps} 站
              </span>

              <div className="flex items-center gap-1.5 text-xs text-slate-500">
                <span>速度</span>
                <select
                  value={speed}
                  onChange={(e) => setSpeed(Number(e.target.value))}
                  className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-xs"
                >
                  <option value={2000}>慢</option>
                  <option value={1200}>中</option>
                  <option value={600}>快</option>
                </select>
              </div>
            </div>
          </div>

          {/* Sidebar timelines */}
          <div className="flex max-h-[720px] flex-col gap-4 overflow-y-auto pr-1">
            <div className="rounded-2xl border border-dashed border-indigo-200 bg-indigo-50/60 p-3 text-xs leading-relaxed text-indigo-700">
              💡 点击右侧时间线中的任意站点，地图会自动定位到该签到位置；拖动下方进度条或点击"播放"可按时间顺序重放该用户当天的移动轨迹。
            </div>
            {selected.map((u) => (
              <UserTimeline
                key={u.userIdx}
                user={u}
                revealCount={revealCount}
                onSelectStop={(lat, lon) => setFocusPoint({ lat, lon })}
              />
            ))}
          </div>
        </div>

        <footer className="mt-8 border-t border-slate-200 pt-4 text-center text-xs text-slate-400">
          Figure 3 — User Trajectory Map · 数据字段对齐 train_encoded.csv（user_idx / poi_idx / category_idx /
          hour_sin / hour_cos / weekday / is_weekend）
        </footer>
      </main>
    </div>
  );
}

function StatCard({ label, value, accent }: { label: string; value: string; accent: string }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white/80 px-4 py-3 shadow-sm">
      <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400">{label}</p>
      <p className={`mt-1 text-lg font-bold ${accent}`}>{value}</p>
    </div>
  );
}
