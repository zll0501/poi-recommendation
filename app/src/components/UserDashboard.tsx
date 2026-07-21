import type { PoiLookupEntry, UserTrajectory } from "../data/trajectories";
import type { UserRecommendationEntry } from "../data/recommendations";
import UserPanel from "./UserPanel";

interface Props {
  user: UserTrajectory | null;
  recommendation: UserRecommendationEntry | null;
  revealCount: number;
  onFocusPoint: (point: { lat: number; lon: number } | null) => void;
}

export default function UserDashboard({ user, recommendation, revealCount, onFocusPoint }: Props) {
  if (!user || !recommendation) {
    return null;
  }

  return (
    <section>
      <UserPanel user={user} recommendation={recommendation} revealCount={revealCount} onFocusPoint={onFocusPoint} />
    </section>
  );
}