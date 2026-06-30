import React, { useMemo, useState } from "react";
import {
  ComposedChart, Area, Line, XAxis, YAxis, ReferenceLine, Tooltip, ResponsiveContainer,
} from "recharts";

// --- palette: a quiet "overcast dawn" — care, not clinical -------------------
const C = {
  paper: "#ECEEEA",
  card: "#F6F7F4",
  ink: "#22312F",
  inkSoft: "#5B6B66",
  band: "#AEC3B6",       // "your normal" band
  bandFill: "#C9D8CD",
  line: "#2F4858",
  amber: "#C2853B",      // deviation accent (not alarm-red)
  care: "#B5543F",       // crisis/support — warm clay
  careBg: "#F4E7E0",
  hair: "#D8DCD5",
};

// --- mock 30 days of wellbeing (0..1), with a dip + partial recovery ---------
const SERIES = (() => {
  const base = [
    .72,.68,.75,.70,.66,.71,.74,.69,.73,.70,.67,.72,.71,.68,
    .60,.55,.48,.44,.41,.38,.43,.40,.46,.52,.58,.61,.59,.63,.66,.64,
  ];
  // baseline from the first 14 stable days
  const warm = base.slice(0, 14).slice().sort((a, b) => a - b);
  const med = warm[Math.floor(warm.length / 2)];
  const mad = Math.max(0.05, warm.map(v => Math.abs(v - med)).sort((a, b) => a - b)[Math.floor(warm.length / 2)]);
  const spread = mad * 1.4826 * 1.5;
  return base.map((w, i) => ({
    day: i + 1,
    wellbeing: +w.toFixed(3),
    lo: +Math.max(0, med - spread).toFixed(3),
    hi: +Math.min(1, med + spread).toFixed(3),
    band: [+Math.max(0, med - spread).toFixed(3), +Math.min(1, med + spread).toFixed(3)],
  }));
})();

const TRIGGERS = [
  { tag: "deadlines", n: 9, mood: "down" },
  { tag: "poor sleep", n: 7, mood: "down" },
  { tag: "skipped meals", n: 5, mood: "down" },
  { tag: "time outdoors", n: 6, mood: "up" },
  { tag: "called a friend", n: 4, mood: "up" },
];

const RESOURCES = [
  { name: "Tele-MANAS (India)", line: "14416", note: "Free · 24/7 · 20+ languages", href: "tel:14416" },
  { name: "988 Lifeline (US)", line: "988", note: "Call or text · 24/7 · confidential", href: "tel:988" },
  { name: "Find A Helpline", line: "findahelpline.com", note: "Verified lines, 130+ countries", href: "https://findahelpline.com" },
];

export default function App() {
  const [showSupport, setShowSupport] = useState(false);
  const [contactOn, setContactOn] = useState(false);

  const last = SERIES[SERIES.length - 1];
  const med = SERIES[0].band[0] + (SERIES[0].band[1] - SERIES[0].band[0]) / 2;
  const z = useMemo(() => {
    const spread = (SERIES[0].hi - SERIES[0].lo) / 2 || 0.05;
    return (last.wellbeing - med) / spread;
  }, [last, med]);
  const severity = z <= -1.5 ? "notable" : z <= -0.8 ? "mild" : "steady";

  return (
    <div style={{ background: C.paper, color: C.ink, minHeight: "100%", fontFamily: "Inter, system-ui, sans-serif" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');
        @media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}
        :focus-visible{outline:2px solid ${C.line};outline-offset:2px;border-radius:6px}`}</style>

      <div className="mx-auto px-5 py-8" style={{ maxWidth: 880 }}>
        {/* header */}
        <div className="flex items-baseline justify-between mb-1">
          <h1 style={{ fontFamily: "Fraunces, serif", fontWeight: 600, fontSize: 30, letterSpacing: "-0.01em" }}>
            Your inner weather
          </h1>
          <span style={{ color: C.inkSoft, fontSize: 13 }}>Day 30 · today</span>
        </div>
        <p style={{ color: C.inkSoft, fontSize: 14, marginBottom: 22 }}>
          Thirty quiet check-ins. The shaded band is what steady looks like <em>for you</em> — not a target, just your own normal.
        </p>

        {/* signature: mood ribbon with the personal baseline band */}
        <div className="rounded-2xl p-4 mb-5" style={{ background: C.card, border: `1px solid ${C.hair}` }}>
          <div className="flex items-center justify-between mb-2 px-1">
            <span style={{ fontSize: 13, color: C.inkSoft }}>Wellbeing index</span>
            <span style={{ fontSize: 13, color: severity === "notable" ? C.amber : C.inkSoft }}>
              {severity === "notable" ? "below your normal lately" : severity === "mild" ? "a little under" : "within your normal"}
            </span>
          </div>
          <div style={{ width: "100%", height: 230 }}>
            <ResponsiveContainer>
              <ComposedChart data={SERIES} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
                <defs>
                  <linearGradient id="bandG" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={C.bandFill} stopOpacity={0.55} />
                    <stop offset="100%" stopColor={C.bandFill} stopOpacity={0.2} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="day" tick={{ fontSize: 11, fill: C.inkSoft }} tickLine={false}
                  axisLine={{ stroke: C.hair }} interval={4} />
                <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tick={{ fontSize: 11, fill: C.inkSoft }}
                  tickLine={false} axisLine={false} tickFormatter={(v) => v === 0.5 ? "ok" : v === 1 ? "good" : "low"} />
                <Tooltip
                  contentStyle={{ background: C.card, border: `1px solid ${C.hair}`, borderRadius: 10, fontSize: 12 }}
                  formatter={(v, n) => n === "wellbeing" ? [Math.round(v * 100) / 100, "wellbeing"] : null}
                  labelFormatter={(l) => `Day ${l}`} />
                {/* the band = "your normal" */}
                <Area dataKey="band" stroke={C.band} strokeWidth={1} fill="url(#bandG)" isAnimationActive={false} />
                <Line dataKey="wellbeing" stroke={C.line} strokeWidth={2.4} dot={false}
                  isAnimationActive={false} type="monotone" />
                <ReferenceLine x={14} stroke={C.amber} strokeDasharray="3 3"
                  label={{ value: "dip begins", fontSize: 10, fill: C.amber, position: "insideTopRight" }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* row: triggers + suggestion */}
        <div className="grid gap-5 mb-5" style={{ gridTemplateColumns: "1.1fr 1fr" }}>
          <div className="rounded-2xl p-5" style={{ background: C.card, border: `1px solid ${C.hair}` }}>
            <h2 style={{ fontFamily: "Fraunces, serif", fontSize: 17, fontWeight: 600, marginBottom: 4 }}>
              What tends to move it
            </h2>
            <p style={{ fontSize: 12.5, color: C.inkSoft, marginBottom: 14 }}>
              Patterns from your own notes. Correlation, not cause — you know your life best.
            </p>
            {TRIGGERS.map((t) => (
              <div key={t.tag} className="flex items-center gap-3 mb-2.5">
                <span style={{ width: 8, height: 8, borderRadius: 99, background: t.mood === "up" ? C.band : C.amber, flexShrink: 0 }} />
                <span style={{ fontSize: 13.5, width: 120 }}>{t.tag}</span>
                <div className="flex-1 rounded-full" style={{ height: 6, background: C.hair }}>
                  <div className="rounded-full" style={{ height: 6, width: `${t.n * 10}%`, background: t.mood === "up" ? C.band : C.amber }} />
                </div>
                <span style={{ fontSize: 12, color: C.inkSoft, width: 48, textAlign: "right" }}>{t.n} days</span>
              </div>
            ))}
          </div>

          <div className="rounded-2xl p-5 flex flex-col" style={{ background: C.card, border: `1px solid ${C.hair}` }}>
            <h2 style={{ fontFamily: "Fraunces, serif", fontSize: 17, fontWeight: 600, marginBottom: 8 }}>
              A small nudge
            </h2>
            <p style={{ fontSize: 14, lineHeight: 1.55, flex: 1 }}>
              Your check-ins have run below your usual for about a week, and they line up with deadline days and short sleep.
              That's worth being gentle with yourself about. If it would help, talking it through with someone you trust can lighten the load.
            </p>
            <span style={{ fontSize: 12, color: C.inkSoft, marginTop: 12 }}>A suggestion, never a prescription.</span>
          </div>
        </div>

        {/* trusted-contact consent — the right pattern, on by choice */}
        <div className="rounded-2xl p-5 mb-5" style={{ background: C.card, border: `1px solid ${C.hair}` }}>
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 style={{ fontFamily: "Fraunces, serif", fontSize: 17, fontWeight: 600, marginBottom: 4 }}>
                Let one person know, only if you choose
              </h2>
              <p style={{ fontSize: 13.5, color: C.inkSoft, lineHeight: 1.5, maxWidth: 560 }}>
                If you turn this on, and a check-in ever suggests you're really struggling, we'll offer to message one person you pick.
                You'll always see it first and can stop it. Nothing is ever sent silently or behind your back.
              </p>
            </div>
            <button
              onClick={() => setContactOn((v) => !v)}
              aria-pressed={contactOn}
              className="rounded-full"
              style={{ width: 52, height: 30, background: contactOn ? C.band : C.hair, position: "relative", flexShrink: 0, transition: "background .2s" }}>
              <span style={{ position: "absolute", top: 3, left: contactOn ? 25 : 3, width: 24, height: 24, borderRadius: 99, background: "#fff", transition: "left .2s", boxShadow: "0 1px 3px rgba(0,0,0,.2)" }} />
            </button>
          </div>
          {contactOn && (
            <p style={{ fontSize: 12.5, color: C.inkSoft, marginTop: 12, paddingTop: 12, borderTop: `1px solid ${C.hair}` }}>
              On. You'd choose the person and how to reach them, and could turn this off anytime.
            </p>
          )}
        </div>

        {/* support — always reachable; preview the in-crisis takeover */}
        <div className="rounded-2xl p-5" style={{ background: showSupport ? C.careBg : C.card, border: `1px solid ${showSupport ? C.care : C.hair}` }}>
          <div className="flex items-center justify-between">
            <h2 style={{ fontFamily: "Fraunces, serif", fontSize: 17, fontWeight: 600, color: showSupport ? C.care : C.ink }}>
              Talk to someone now
            </h2>
            <button onClick={() => setShowSupport((v) => !v)}
              style={{ fontSize: 12.5, color: C.inkSoft, textDecoration: "underline" }}>
              {showSupport ? "hide" : "preview support view"}
            </button>
          </div>
          <p style={{ fontSize: 13.5, color: showSupport ? C.care : C.inkSoft, marginTop: 6, marginBottom: showSupport ? 16 : 0, lineHeight: 1.5 }}>
            {showSupport
              ? "It sounds like you're carrying a lot right now, and you don't have to do it alone. A trained counsellor can talk with you this minute."
              : "Free, confidential help is always one tap away — you never have to be in crisis to reach out."}
          </p>
          {showSupport && (
            <div className="grid gap-2.5" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
              {RESOURCES.map((r) => (
                <a key={r.name} href={r.href} target="_blank" rel="noreferrer"
                  className="rounded-xl p-3.5 block" style={{ background: "#fff", border: `1px solid ${C.care}33`, textDecoration: "none", color: C.ink }}>
                  <div style={{ fontSize: 13, fontWeight: 600 }}>{r.name}</div>
                  <div style={{ fontSize: 20, fontWeight: 600, color: C.care, fontFamily: "Fraunces, serif", margin: "2px 0" }}>{r.line}</div>
                  <div style={{ fontSize: 11.5, color: C.inkSoft }}>{r.note}</div>
                </a>
              ))}
            </div>
          )}
        </div>

        <p style={{ fontSize: 11.5, color: C.inkSoft, marginTop: 18, textAlign: "center", lineHeight: 1.5 }}>
          This is a wellbeing companion, not a medical device, diagnosis, or a substitute for professional care.
          In an emergency, contact local emergency services.
        </p>
      </div>
    </div>
  );
}
