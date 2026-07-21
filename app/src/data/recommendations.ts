import rawRecommendations from "./user_recommendations.json";

export interface RecommendationItem {
  rank: number;
  poiIdx: number;
  poiId: string;
  categoryName: string;
  latitude: number;
  longitude: number;
  trainingVisitCount: number;
}

export interface UserRecommendationEntry {
  userId: number;
  userIdx: number;
  latestEventId: number;
  eventCount: number;
  topK: RecommendationItem[];
}

interface RecommendationFile {
  generatedAt?: string;
  source?: string;
  users: UserRecommendationEntry[];
}

const recommendationFile = rawRecommendations as RecommendationFile;

export const recommendationEntries: UserRecommendationEntry[] = recommendationFile.users ?? [];

export function getRecommendationForUserId(userId: number): UserRecommendationEntry | null {
  return recommendationEntries.find((entry) => entry.userId === userId) ?? null;
}

export function getRecommendationForTrajectoryUser(userId?: number, userIdx?: number): UserRecommendationEntry | null {
  if (typeof userId === "number") {
    const matched = getRecommendationForUserId(userId);
    if (matched) return matched;
  }

  if (typeof userIdx === "number") {
    return recommendationEntries.find((entry) => entry.userIdx === userIdx) ?? null;
  }

  return null;
}