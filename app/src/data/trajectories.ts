export interface CheckIn {
  eventId: number;
  order: number;
  poiIdx: number;
  poiName: string;
  category: string;
  emoji: string;
  lat: number;
  lon: number;
  weekday: string;
  hour: number;
  timeLabel: string;
}

export interface UserTrajectory {
  userId: number;
  userIdx: number;
  userLabel: string;
  borough: string;
  color: string;
  weekday: string;
  predictionEventId: number;
  checkins: CheckIn[];
  targetCheckin: CheckIn;
}

export interface TrajectoryViewData {
  generatedAt?: string;
  source?: string;
  requestedSampleCount?: number;
  unavailableUserIds?: number[];
  users: UserTrajectory[];
}

export async function loadTrajectoryViewData(): Promise<UserTrajectory[]> {
  const response = await fetch("/data/trajectories_view.json");
  if (!response.ok) {
    throw new Error(`轨迹数据请求失败：HTTP ${response.status}`);
  }

  const data = (await response.json()) as TrajectoryViewData;
  return data.users ?? [];
}

export function pickRandomUser(users: UserTrajectory[]): UserTrajectory | null {
  if (users.length === 0) return null;
  const index = Math.floor(Math.random() * users.length);
  return users[index] ?? null;
}

// 计算两点间球面距离（km），用于展示统计信息。
export function haversineKm(a: CheckIn, b: CheckIn): number {
  const radius = 6371;
  const dLat = ((b.lat - a.lat) * Math.PI) / 180;
  const dLon = ((b.lon - a.lon) * Math.PI) / 180;
  const lat1 = (a.lat * Math.PI) / 180;
  const lat2 = (b.lat * Math.PI) / 180;
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.sin(dLon / 2) ** 2 * Math.cos(lat1) * Math.cos(lat2);
  return radius * 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
}

export function totalDistance(checkins: CheckIn[]): number {
  let sum = 0;
  for (let index = 1; index < checkins.length; index += 1) {
    sum += haversineKm(checkins[index - 1], checkins[index]);
  }
  return sum;
}
