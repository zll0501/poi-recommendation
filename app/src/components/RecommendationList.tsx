import type { UserRecommendationEntry } from "../data/recommendations";

interface Props {
  recommendation: UserRecommendationEntry;
  highlightPoiIdx?: number | null;
  onPickPoi?: (lat: number, lon: number) => void;
}

export default function RecommendationList({
  recommendation,
  highlightPoiIdx,
  onPickPoi,
}: Props) {
  return (
    <div className="rounded-[24px] border border-slate-200 bg-slate-50/70 p-4">
      <div className="flex items-end justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-amber-500">Top-10 推荐</p>
          <h4 className="mt-1 text-lg font-semibold text-slate-900">推荐结果排行榜</h4>
        </div>
        <div className="text-right text-xs text-slate-400">
          <p>样本事件 {recommendation.eventId}</p>
        </div>
      </div>

      <div className="mt-3 space-y-2">
        {recommendation.topK.map((item) => {
          const isHit = highlightPoiIdx === item.poiIdx;
          return (
            <button
              key={`${recommendation.userId}-${recommendation.latestEventId}-${item.rank}-${item.poiIdx}`}
              onClick={() => {
                if (onPickPoi) onPickPoi(item.latitude, item.longitude);
              }}
              className={`flex w-full items-center gap-3 rounded-2xl border px-3 py-2 text-left transition ${
                isHit
                  ? "border-emerald-300 bg-emerald-50 shadow-sm"
                  : "border-transparent bg-white hover:border-slate-200 hover:bg-slate-50"
              }`}
            >
              <span
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-bold ${
                  item.rank === 1
                    ? "bg-amber-500 text-white"
                    : item.rank <= 3
                      ? "bg-amber-400/90 text-white"
                      : "bg-slate-200 text-slate-700"
                }`}
              >
                {item.rank}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium text-slate-800">
                  📍 {item.categoryName}
                </span>
                <span className="block truncate text-xs text-slate-400">
                  POI ID: {item.poiId} · Visits: {item.trainingVisitCount}
                </span>
              </span>
              {isHit ? (
                <span className="shrink-0 rounded-full bg-emerald-600 px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-white">
                  Hit
                </span>
              ) : null}
            </button>
          );
        })}
      </div>
    </div>
  );
}