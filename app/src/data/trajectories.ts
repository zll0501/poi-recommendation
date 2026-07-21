export interface CheckIn {
  order: number;
  poiIdx: number;
  poiName: string;
  category: string;
  emoji: string;
  lat: number;
  lon: number;
  weekday: string;
  hour: number;
  timeLabel: string; // e.g. "08:10"
}

export interface UserTrajectory {
  userId?: number;
  userIdx: number;
  userLabel: string;
  borough: string;
  color: string;
  weekday: string;
  checkins: CheckIn[];
}

export interface TrajectoryViewData {
  generatedAt?: string;
  source?: string;
  users: UserTrajectory[];
}

export interface PoiLookupEntry {
  poiIdx: number;
  poiName: string;
  category: string;
  emoji: string;
  lat: number;
  lon: number;
  visitCount: number;
}

export const USER_COLORS = [
  "#e11d48", // rose
  "#2563eb", // blue
  "#059669", // emerald
  "#d97706", // amber
  "#7c3aed", // violet
  "#db2777", // pink
];

function getEmoji(category: string) {
  const normalized = category.toLowerCase();
  if (normalized.includes("home")) return "🏠";
  if (normalized.includes("subway") || normalized.includes("station")) return "🚇";
  if (normalized.includes("coffee") || normalized.includes("cafe") || normalized.includes("café")) return "☕";
  if (normalized.includes("office")) return "💼";
  if (normalized.includes("restaurant") || normalized.includes("diner")) return "🍽️";
  if (normalized.includes("food truck")) return "🚚";
  if (normalized.includes("gym")) return "🏋️";
  if (normalized.includes("park")) return "🌳";
  if (normalized.includes("bar") || normalized.includes("pub")) return "🍸";
  if (normalized.includes("mall") || normalized.includes("shopping")) return "🛍️";
  if (normalized.includes("bridge")) return "🌉";
  if (normalized.includes("school") || normalized.includes("university")) return "🏫";
  if (normalized.includes("grocery") || normalized.includes("supermarket")) return "🛒";
  if (normalized.includes("museum") || normalized.includes("art")) return "🎨";
  if (normalized.includes("cinema") || normalized.includes("movie")) return "🎬";
  return "📍";
}

function formatTimeLabel(hour: number, localTime?: string) {
  if (localTime) {
    const match = localTime.match(/(\d{2}:\d{2})/);
    if (match) return match[1];
  }
  return `${String(hour).padStart(2, "0")}:00`;
}

function deriveBorough(lat: number, lon: number) {
  if (lat >= 40.79) return lon < -73.95 ? "Bronx / Upper Manhattan" : "Queens / Bronx";
  if (lat >= 40.73) return lon < -74.0 ? "Manhattan · West Side" : "Manhattan · East Side";
  if (lat >= 40.66) return lon < -73.95 ? "Brooklyn" : "Queens";
  return "NYC";
}

export function buildTrajectoryViewData(
  rows: Array<Record<string, string>>,
  limitUsers = 6
): UserTrajectory[] {
  const grouped = new Map<number, Array<Record<string, string>>>();

  for (const row of rows) {
    const userIdx = Number(row.user_idx);
    if (!Number.isFinite(userIdx)) continue;
    const existing = grouped.get(userIdx) ?? [];
    existing.push(row);
    grouped.set(userIdx, existing);
  }

  return [...grouped.entries()]
    .map(([userIdx, items]) => ({
      userIdx,
      items: items.sort((a, b) => Number(a.timestamp) - Number(b.timestamp)),
    }))
    .sort((a, b) => b.items.length - a.items.length)
    .slice(0, limitUsers)
    .map(({ userIdx, items }, index) => {
      const first = items[0];
      const weekday = first.weekday || "";

      return {
        userId: first.user_id ? Number(first.user_id) : undefined,
        userIdx,
        userLabel: first.user_id ? `User ${first.user_id}` : `User #${userIdx}`,
        borough: `${deriveBorough(Number(first.latitude), Number(first.longitude))} · ${weekday || "Trajectory"}`,
        color: USER_COLORS[index % USER_COLORS.length],
        weekday: weekday || "",
        checkins: items.map((row, itemIndex) => ({
          order: itemIndex + 1,
          poiIdx: Number(row.poi_idx),
          poiName: row.category_name || `POI ${row.poi_idx}`,
          category: row.category_name || "Unknown",
          emoji: getEmoji(row.category_name || ""),
          lat: Number(row.latitude),
          lon: Number(row.longitude),
          weekday: row.weekday || weekday || "",
          hour: Number(row.hour),
          timeLabel: formatTimeLabel(Number(row.hour), row.local_time),
        })),
      };
    });
}

export async function loadTrajectoryViewData(): Promise<UserTrajectory[]> {
  const response = await fetch("/data/trajectories_view.json");
  if (!response.ok) return [];

  const data = (await response.json()) as TrajectoryViewData;
  return data.users ?? [];
}

export function pickRandomUsers(users: UserTrajectory[], min = 1, max = 3): UserTrajectory[] {
  if (users.length === 0) return [];

  const count = Math.floor(Math.random() * (max - min + 1)) + min;
  const shuffled = [...users].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(count, users.length));
}

export function pickRandomUser(users: UserTrajectory[]): UserTrajectory | null {
  if (users.length === 0) return null;
  return pickRandomUsers(users, 1, 1)[0] ?? null;
}

// 计算两点间球面距离 (km)，用于展示统计信息
export function haversineKm(a: CheckIn, b: CheckIn): number {
  const R = 6371;
  const dLat = ((b.lat - a.lat) * Math.PI) / 180;
  const dLon = ((b.lon - a.lon) * Math.PI) / 180;
  const lat1 = (a.lat * Math.PI) / 180;
  const lat2 = (b.lat * Math.PI) / 180;
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.sin(dLon / 2) ** 2 * Math.cos(lat1) * Math.cos(lat2);
  return R * 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
}

export function totalDistance(checkins: CheckIn[]): number {
  let sum = 0;
  for (let i = 1; i < checkins.length; i++) {
    sum += haversineKm(checkins[i - 1], checkins[i]);
  }
  return sum;
}

export function buildPoiLookup(users: UserTrajectory[]): Map<number, PoiLookupEntry> {
  const lookup = new Map<number, PoiLookupEntry>();

  for (const user of users) {
    for (const checkin of user.checkins) {
      const existing = lookup.get(checkin.poiIdx);
      if (existing) {
        existing.visitCount += 1;
        continue;
      }

      lookup.set(checkin.poiIdx, {
        poiIdx: checkin.poiIdx,
        poiName: checkin.poiName,
        category: checkin.category,
        emoji: checkin.emoji,
        lat: checkin.lat,
        lon: checkin.lon,
        visitCount: 1,
      });
    }
  }

  return lookup;
}
