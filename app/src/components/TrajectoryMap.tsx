import { Fragment, useEffect, useMemo } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Polyline,
  Popup,
  Tooltip,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { UserTrajectory } from "../data/trajectories";

function divIcon(label: string, color: string, active: boolean, big: boolean) {
  const size = big ? 34 : 26;
  return L.divIcon({
    className: "custom-poi-icon",
    html: `<div style="
        width:${size}px;height:${size}px;border-radius:9999px;
        background:${color};
        display:flex;align-items:center;justify-content:center;
        color:white;font-weight:700;font-size:${big ? 14 : 12}px;
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

export default function TrajectoryMap({ users, revealCount, focusPoint }: Props) {
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
        const visible = u.checkins.slice(0, Math.max(1, Math.min(revealCount, u.checkins.length)));
        const positions = visible.map((c) => [c.lat, c.lon] as [number, number]);
        return (
          <Fragment key={u.userIdx}>
            <Polyline
              positions={positions}
              pathOptions={{
                color: u.color,
                weight: 3.5,
                opacity: 0.85,
                dashArray: "1 8",
                lineCap: "round",
              }}
            />
            {visible.map((c, idx) => {
              const isFirst = idx === 0;
              const isLast = idx === visible.length - 1 && visible.length === u.checkins.length;
              const label = isFirst ? "起" : isLast ? "终" : String(c.order);
              return (
                <Marker
                  key={`${u.userIdx}-${c.order}`}
                  position={[c.lat, c.lon]}
                  icon={divIcon(label, u.color, idx === visible.length - 1, isFirst || isLast)}
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
              );
            })}
          </Fragment>
        );
      })}
    </MapContainer>
  );
}
