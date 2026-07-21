import { useEffect, useMemo, useState } from "react";
import TopControls from "./components/TopControls";
import TrajectoryMap from "./components/TrajectoryMap";
import UserDashboard from "./components/UserDashboard";
import RecommendationList from "./components/RecommendationList";
import {
  loadTrajectoryViewData,
  pickRandomUser,
  totalDistance,
  type UserTrajectory,
} from "./data/trajectories";
import { getRecommendationForTrajectoryUser } from "./data/recommendations";
import sampleUsers from "./data/sample_users.json";

export default function App() {
  const [availableUsers, setAvailableUsers] = useState<UserTrajectory[]>([]);
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null);
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
        const displayableUsers = users.filter((user) =>
          getRecommendationForTrajectoryUser(user.userId, user.userIdx)
        );
        if (displayableUsers.length === 0) {
          throw new Error("样本池中没有同时具备轨迹和推荐结果的用户");
        }
        setAvailableUsers(displayableUsers);
        setSelectedUserId(resolveInitialSelection(displayableUsers));
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

  const selectedUser = useMemo(() => {
    if (selectedUserId === null) return null;
    return (
      availableUsers.find((user) => user.userId === selectedUserId) ??
      availableUsers.find((user) => user.userIdx === selectedUserId) ??
      null
    );
  }, [availableUsers, selectedUserId]);

  const selectedRecommendation = useMemo(
    () => getRecommendationForTrajectoryUser(selectedUser?.userId, selectedUser?.userIdx),
    [selectedUser]
  );

  const hitPoiIdx = useMemo(() => {
    if (!selectedUser || !selectedRecommendation) return null;
    return (
      selectedRecommendation.topK.find(
        (item) => item.poiIdx === selectedUser.targetCheckin.poiIdx
      )?.poiIdx ?? null
    );
  }, [selectedRecommendation, selectedUser]);

  const maxSteps = useMemo(
    () => Math.max(1, selectedUser?.checkins.length ?? 1),
    [selectedUser]
  );

  useEffect(() => {
    setRevealCount(maxSteps);
    setPlaying(false);
  }, [selectedUserId, maxSteps]);

  useEffect(() => {
    if (!selectedUser) return;
    if (!playing) return;
    if (revealCount >= maxSteps) {
      setPlaying(false);
      return;
    }

    const timer = setTimeout(() => {
      setRevealCount((current) => Math.min(maxSteps, current + 1));
    }, speed);

    return () => clearTimeout(timer);
  }, [playing, revealCount, maxSteps, selectedUser, speed]);

  const selectUser = (userId: number) => {
    setSelectedUserId(userId);
    setFocusPoint(null);
  };

  const reroll = () => {
    const candidates = availableUsers.filter(
      (user) => (user.userId ?? user.userIdx) !== selectedUserId
    );
    const randomPick = pickRandomUser(candidates.length > 0 ? candidates : availableUsers);
    if (randomPick) setSelectedUserId(randomPick.userId);
    setFocusPoint(null);
  };

  const togglePlay = () => {
    if (revealCount >= maxSteps) setRevealCount(1);
    setPlaying((current) => !current);
  };

  const totalCheckins = selectedUser ? selectedUser.checkins.length : 0;
  const totalKm = selectedUser ? totalDistance(selectedUser.checkins) : 0;

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[radial-gradient(circle_at_top,_rgba(99,102,241,0.12),_transparent_45%),linear-gradient(180deg,#f8fafc_0%,#ffffff_55%,#eef2ff_100%)] text-slate-600">
        正在加载真实轨迹数据…
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[radial-gradient(circle_at_top,_rgba(99,102,241,0.12),_transparent_45%),linear-gradient(180deg,#f8fafc_0%,#ffffff_55%,#eef2ff_100%)] px-6 text-center text-slate-700">
        <div className="max-w-xl rounded-3xl border border-rose-200 bg-white/90 p-6 shadow-sm">
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
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(99,102,241,0.12),_transparent_45%),linear-gradient(180deg,#f8fafc_0%,#ffffff_50%,#eef2ff_100%)] text-slate-900">
      <main className="mx-auto flex min-h-screen w-full max-w-[1600px] flex-col gap-5 px-4 py-4 lg:px-6">
        <TopControls
          availableUsers={availableUsers}
          selectedUser={selectedUser}
          samplePoolSize={sampleUsers.length}
          onRandomize={reroll}
          onSelectUser={selectUser}
          onTogglePlay={togglePlay}
          onRevealChange={(value) => {
            setPlaying(false);
            setRevealCount(value);
          }}
          onSpeedChange={setSpeed}
          playing={playing}
          revealCount={revealCount}
          maxSteps={maxSteps}
          speed={speed}
          totalCheckins={totalCheckins}
          totalDistanceKm={totalKm}
        />

        <section className="overflow-hidden rounded-[28px] border border-slate-200/80 bg-white/80 shadow-[0_24px_80px_rgba(15,23,42,0.08)] backdrop-blur">
          <div className="border-b border-slate-200/80 px-4 py-3 sm:px-6">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-indigo-500">
                  Figure 3 · User Trajectory Visualization
                </p>
                <h1 className="mt-1 text-2xl font-semibold tracking-tight text-slate-900">
                  用户历史签到轨迹地图
                </h1>
                <p className="mt-1 max-w-4xl text-sm leading-relaxed text-slate-500">
                  左侧回放用户最近轨迹，右侧展示同一预测事件的Top-10地点；点击推荐项可在地图中定位。
                </p>
              </div>
              <div className="rounded-2xl bg-slate-50 px-4 py-3 text-right text-xs text-slate-500">
                <p className="font-medium text-slate-700">当前样本池</p>
                <p>{availableUsers.length} 位用户</p>
              </div>
            </div>
          </div>

          <div className="grid xl:grid-cols-[minmax(0,1fr)_390px]">
            <div className="h-[58vh] min-h-[520px] bg-slate-100/50">
              <TrajectoryMap
                users={selectedUser ? [selectedUser] : []}
                recommendation={selectedRecommendation}
                revealCount={revealCount}
                focusPoint={focusPoint}
              />
            </div>

            <aside className="border-t border-slate-200/80 bg-white/90 p-4 xl:h-[58vh] xl:min-h-[520px] xl:overflow-y-auto xl:border-l xl:border-t-0">
              {selectedRecommendation ? (
                <RecommendationList
                  recommendation={selectedRecommendation}
                  highlightPoiIdx={hitPoiIdx}
                  onPickPoi={(lat, lon) => setFocusPoint({ lat, lon })}
                />
              ) : (
                <div className="rounded-[24px] border border-amber-200 bg-amber-50 p-5 text-sm text-amber-800">
                  当前用户没有可展示的推荐结果。
                </div>
              )}
            </aside>
          </div>
        </section>

        <UserDashboard
          user={selectedUser}
          recommendation={selectedRecommendation}
          revealCount={revealCount}
          onFocusPoint={setFocusPoint}
        />
      </main>
    </div>
  );
}

function resolveInitialSelection(users: UserTrajectory[]): number | null {
  if (users.length === 0) return null;

  const matchedIds = sampleUsers
    .map((userId) => users.find((user) => user.userId === userId)?.userId)
    .filter((userId): userId is number => typeof userId === "number");

  if (matchedIds.length >= 1) {
    return matchedIds[0];
  }

  const randomPick = pickRandomUser(users);
  return randomPick?.userId ?? users[0].userId;
}
