import type { UserTrajectory } from "../data/trajectories";
import type { UserRecommendationEntry } from "../data/recommendations";
import UserPanel from "./UserPanel";

interface Props {
  user: UserTrajectory | null;
  recommendation: UserRecommendationEntry | null;
  revealCount: number;
  onFocusPoint: (point: { lat: number; lon: number } | null) => void;
}

export default function UserDashboard({ user, recommendation, revealCount, onFocusPoint }: Props) {
  if (!user) {
    return null;
  }

  if (!recommendation) {
    return (
      <section className="rounded-[28px] border border-amber-200 bg-amber-50 p-6 text-sm text-amber-800">
        当前用户没有可对齐的推荐结果，请重新抽取其他样本用户。
      </section>
    );
  }

  return (
    <section>
      <UserPanel user={user} recommendation={recommendation} revealCount={revealCount} onFocusPoint={onFocusPoint} />
    </section>
  );
}
