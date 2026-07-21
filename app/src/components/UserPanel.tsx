import { useMemo } from "react";
import CategoryBarChart from "./CategoryBarChart";
import InterestRadarChart from "./InterestRadarChart";
import RecommendationList from "./RecommendationList";
import UserTimeline from "./UserTimeline";
import type { UserTrajectory } from "../data/trajectories";
import type { UserRecommendationEntry } from "../data/recommendations";
import { totalDistance } from "../data/trajectories";

interface Props {
  user: UserTrajectory;
  recommendation: UserRecommendationEntry;
  revealCount: number;
  onFocusPoint: (point: { lat: number; lon: number } | null) => void;
}

export default function UserPanel({ user, recommendation, revealCount, onFocusPoint }: Props) {
  const totalKm = totalDistance(user.checkins);
  const visibleCount = Math.max(1, Math.min(revealCount, user.checkins.length));
  const targetCheckin = user.targetCheckin;

  const categoryStats = useMemo(() => {
    const counts = new Map<string, number>();
    for (const checkin of user.checkins) {
      counts.set(checkin.category, (counts.get(checkin.category) ?? 0) + 1);
    }

    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([category, count]) => ({ category, count }));
  }, [user]);

  const radarCategories = useMemo(() => {
    if (categoryStats.length === 0) return ["无数据", "无数据"];
    return [...categoryStats.map((item) => item.category), categoryStats[0].category];
  }, [categoryStats]);

  const radarCounts = useMemo(() => {
    if (categoryStats.length === 0) return [0, 0];
    return [...categoryStats.map((item) => item.count), categoryStats[0].count];
  }, [categoryStats]);

  const radarPercentages = useMemo(() => {
    const total = radarCounts.slice(0, -1).reduce((sum, value) => sum + value, 0);
    if (total <= 0) return [0, 0];
    return radarCounts.map((value) => Number(((value / total) * 100).toFixed(1)));
  }, [radarCounts]);

  const topCategories = categoryStats.length === 0 ? ["无数据"] : categoryStats.map((item) => item.category);
  const topCounts = categoryStats.length === 0 ? [0] : categoryStats.map((item) => item.count);

  const hitPoiIdx =
    recommendation.topK.find((item) => item.poiIdx === targetCheckin.poiIdx)?.poiIdx ?? null;

  return (
    <section className="rounded-[28px] border border-slate-200/80 bg-white/85 p-4 shadow-[0_18px_50px_rgba(15,23,42,0.05)] backdrop-blur sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-indigo-500">User Panel</p>
          <h3 className="mt-1 text-xl font-semibold tracking-tight text-slate-900">{user.userLabel}</h3>
          <p className="mt-1 text-sm text-slate-500">{user.borough}</p>
        </div>

        <div className="grid grid-cols-2 gap-2 text-right text-xs text-slate-500">
          <MiniStat label="签到次数" value={String(user.checkins.length)} />
          <MiniStat label="总里程" value={`${totalKm.toFixed(1)} km`} />
          <MiniStat label="当前进度" value={`${visibleCount}/${user.checkins.length}`} />
          <MiniStat label="预测事件" value={String(user.predictionEventId)} />
        </div>
      </div>

      <div className="mt-4 rounded-[24px] bg-slate-50/70 p-4">
        <div className="flex items-end justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-500">基础名片</p>
            <p className="mt-1 text-sm text-slate-500">
              真实下一站：{targetCheckin.emoji} {targetCheckin.poiName}
            </p>
          </div>
          <span className="rounded-full bg-emerald-600 px-3 py-1 text-xs font-semibold text-white">
            Ground Truth
          </span>
        </div>

        <div className="mt-4 grid gap-4 xl:grid-cols-[1fr_1.1fr]">
          <div className="min-h-[320px] rounded-[24px] bg-white p-2 shadow-sm">
            <InterestRadarChart
              userId={user.userIdx}
              categories={radarCategories}
              counts={radarCounts}
              percentages={radarPercentages}
            />
          </div>

          <div className="min-h-[320px] rounded-[24px] bg-white p-2 shadow-sm">
            <CategoryBarChart
              title="POI 类别统计"
              categories={topCategories.slice().reverse()}
              counts={topCounts.slice().reverse()}
            />
          </div>
        </div>
      </div>

      <div className="mt-4">
        <RecommendationList
          recommendation={recommendation}
          highlightPoiIdx={hitPoiIdx}
          onPickPoi={(lat, lon) => onFocusPoint({ lat, lon })}
        />
      </div>

      <div className="mt-4 rounded-[24px] border border-slate-200 bg-slate-50/70 p-4">
        <div className="mb-3 flex items-end justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-indigo-500">轨迹列表</p>
            <p className="mt-1 text-sm text-slate-500">点击任意站点即可在地图上定位</p>
          </div>
        </div>
        <UserTimeline
          user={user}
          revealCount={revealCount}
          onSelectStop={(lat, lon) => onFocusPoint({ lat, lon })}
        />
      </div>
    </section>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-slate-50 px-3 py-2 text-left">
      <p className="text-[10px] font-medium uppercase tracking-wide text-slate-400">{label}</p>
      <p className="mt-1 text-sm font-semibold text-slate-800">{value}</p>
    </div>
  );
}
