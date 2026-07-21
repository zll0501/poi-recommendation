import { Fragment, useEffect, useMemo } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Polyline,
  Popup,
  Tooltip,
  CircleMarker,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { UserTrajectory } from "../data/trajectories";
import type { UserRecommendationEntry } from "../data/recommendations";

function trailIcon(label: string, color: string, active: boolean, size: number) {
  return L.divIcon({
    className: "custom-poi-icon",
    html: `<div style="
        width:${size}px;height:${size}px;border-radius:9999px;
        background:${color};
        display:flex;align-items:center;justify-content:center;
        color:white;font-weight:700;font-size:${Math.max(10, size * 0.4)}px;
        border:2px solid white;
        box-shadow:0 2px 8px rgba(0,0,0,0.35);
        transform:scale(${active ? 1 : 0.9});
        opacity:${active ? 1 : 0.55};
        transition: all .25s ease;
      ">${label}</div>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

interface Props {
  users: UserTrajectory[];
  revealCount: number; // 每个用户已展示到第几个签到点（含），用于动画播放
  focusPoint?: { lat: number; lon: number } | null;
  recommendation: UserRecommendationEntry | null;
}

function FitBounds({ users }: { users: UserTrajectory[] }) {
  const map = useMap();
  useEffect(() => {
    const pts = users.flatMap((u) => u.checkins.map((c) => [c.lat, c.lon] as [number, number]));
    if (pts.length === 0) return;
    if (pts.length === 1) {
      map.setView(pts[0], 14);
    } else {
      map.fitBounds(L.latLngBounds(pts), { padding: [60, 60] });
    }
  }, [users, map]);
  return null;
}

function FlyToPoint({ point }: { point?: { lat: number; lon: number } | null }) {
  const map = useMap();
  useEffect(() => {
    if (point) {
      map.flyTo([point.lat, point.lon], Math.max(map.getZoom(), 14), { duration: 0.8 });
    }
  }, [point, map]);
  return null;
}

function recommendationIcon(rank: number) {
  const size = Math.max(18, 30 - rank * 1.2);
  const opacity = Math.max(0.55, 1 - (rank - 1) * 0.05);
  return L.divIcon({
    className: "custom-recommendation-icon",
    html: `<div style="
      width:${size}px;height:${size}px;border-radius:9999px;
      background:rgba(245,158,11,${opacity});
      color:white;display:flex;align-items:center;justify-content:center;
      border:2px solid rgba(255,255,255,0.95);
      box-shadow:0 6px 14px rgba(245,158,11,0.28);
      font-size:${Math.max(9, size * 0.35)}px;
      font-weight:800;
    ">${rank}</div>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

function groundTruthIcon() {
  return L.divIcon({
    className: "ground-truth-icon",
    html: `<div class="ground-truth-marker">★</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  });
}

export default function TrajectoryMap({ users, revealCount, focusPoint, recommendation }: Props) {
  const center = useMemo<[number, number]>(() => [40.7549, -73.99], []);

  return (
    <MapContainer
      center={center}
      zoom={11}
      scrollWheelZoom
      className="h-full w-full rounded-2xl"
      style={{ background: "#e5e7eb" }}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; CARTO'
        url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
      />
      <FitBounds users={users} />
      <FlyToPoint point={focusPoint} />

      {users.map((u) => {
        const visibleCount = Math.max(1, Math.min(revealCount, u.checkins.length));
        const visible = u.checkins.slice(0, visibleCount);
        const tailStart = Math.max(0, visible.length - 15);
        const tail = visible.slice(tailStart);
        const lastPoint = visible[visible.length - 1];
        const nextCheckin = u.checkins[visibleCount] ?? null;
        const hitRecommendation = nextCheckin
          ? recommendation?.topK.find((item) => item.poiIdx === nextCheckin.poiIdx) ?? null
          : null;

        return (
          <Fragment key={u.userIdx}>
            {u.checkins.slice(0, tailStart).map((c) => (
              <CircleMarker
                key={`${u.userIdx}-${c.order}-ghost`}
                center={[c.lat, c.lon]}
                radius={3}
                pathOptions={{
                  color: u.color,
                  fillColor: u.color,
                  fillOpacity: 0.1,
                  opacity: 0.12,
                  weight: 1,
                }}
              />
            ))}

            {tail.map((c, idx) => {
              const tailIndex = tailStart + idx;
              const previous = tailIndex > 0 ? visible[tailIndex - 1] : null;
              const segmentOpacity = Math.max(0.25, 0.35 + (idx / Math.max(1, tail.length - 1)) * 0.55);
              const isLast = tailIndex === visible.length - 1;

              return (
                <Fragment key={`${u.userIdx}-${c.order}-tail`}>
                  {previous ? (
                    <Polyline
                      positions={[
                        [previous.lat, previous.lon],
                        [c.lat, c.lon],
                      ]}
                      pathOptions={{
                        color: u.color,
                        weight: isLast ? 5 : 3.5,
                        opacity: segmentOpacity,
                        lineCap: "round",
                      }}
                    />
                  ) : null}
                  <Marker
                    position={[c.lat, c.lon]}
                    icon={trailIcon(String(c.order), u.color, isLast, isLast ? 30 : 18)}
                  >
                    <Tooltip direction="top" offset={[0, -14]} opacity={0.95}>
                      <span className="font-medium">{u.userLabel}</span>
                    </Tooltip>
                    <Popup>
                      <div className="space-y-1 text-sm">
                        <div className="font-semibold text-slate-800">
                          {c.emoji} {c.poiName}
                        </div>
                        <div className="text-slate-500">{c.category}</div>
                        <div className="text-slate-500">
                          第 {c.order} 站 · {c.weekday} {c.timeLabel}
                        </div>
                        <div className="text-slate-400">{u.userLabel}</div>
                      </div>
                    </Popup>
                  </Marker>
                </Fragment>
              );
            })}

            {nextCheckin ? (
              <Marker position={[nextCheckin.lat, nextCheckin.lon]} icon={groundTruthIcon()}>
                <Tooltip direction="top" offset={[0, -10]} opacity={0.96} permanent>
                  <span className="ground-truth-tooltip">Ground Truth</span>
                </Tooltip>
                <Popup>
                  <div className="space-y-1 text-sm">
                    <div className="font-semibold text-emerald-700">
                      {nextCheckin.emoji} {nextCheckin.poiName}
                    </div>
                    <div className="text-slate-500">真实下一 POI</div>
                  </div>
                </Popup>
              </Marker>
            ) : null}

            {recommendation?.topK.map((item) => {
              const hit = hitRecommendation?.poiIdx === item.poiIdx;

              return (
                <Fragment key={`${u.userIdx}-rec-${item.rank}-${item.poiIdx}`}>
                  {hit && lastPoint ? (
                    <Polyline
                      positions={[
                        [lastPoint.lat, lastPoint.lon],
                        [item.latitude, item.longitude],
                      ]}
                      pathOptions={{
                        color: "#10b981",
                        weight: 3,
                        opacity: 0.9,
                        dashArray: "7 8",
                        lineCap: "round",
                      }}
                    />
                  ) : null}
                  <Marker position={[item.latitude, item.longitude]} icon={recommendationIcon(item.rank)}>
                    <Tooltip direction="top" offset={[0, -12]} opacity={0.96}>
                      <span className="font-medium text-slate-700">
                        Rank {item.rank}{hit ? " · Hit!" : ""}
                      </span>
                    </Tooltip>
                    <Popup>
                      <div className="space-y-1 text-sm">
                        <div className="font-semibold text-amber-700">
                          📍 {item.categoryName}
                        </div>
                        <div className="text-slate-500">POI ID: {item.poiId}</div>
                        <div className="text-slate-500">Rank {item.rank}</div>
                        <div className="text-slate-500">Training Visits: {item.trainingVisitCount}</div>
                        {hit ? <div className="font-semibold text-emerald-600">命中真实下一 POI</div> : null}
                      </div>
                    </Popup>
                  </Marker>
                </Fragment>
              );
            })}
          </Fragment>
        );
      })}
    </MapContainer>
  );
}
