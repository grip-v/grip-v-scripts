"""Robot Road Trip (after Jane Street, Jul 2025): robot cars with different
top speeds drive a one-lane road with a shoulder. Cars never slow down in
the lane; when a faster car catches up with a slower one, the slower car
pulls onto the shoulder and waits until the faster car is past and the lane
is clear, then pulls back in at full speed.

The simulation steps at DT, but every car only ever drives at its own speed
or stands still, so its whole trip is stored as an exact event log (enter /
pull over / rejoin / finish) that positions can be rebuilt from."""
from __future__ import annotations
import random

ROAD = 12.0
CAR = 0.5                  # car length (units)
CATCH_GAP = 0.12           # a faster car this close behind forces the slower one onto the shoulder
REJOIN_AHEAD = 0.35        # the passer must be this far ahead before the waiting car pulls back in
REJOIN_BEHIND = 1.2        # ... and no lane car within this distance behind it
DT = 0.005
CARS = [("A", "#E4553F"), ("B", "#3F8FE4"), ("C", "#3FAF5B"), ("D", "#D99A1F"), ("E", "#9B5FE4"), ("F", "#2FB5B0")]


def simulate(cars: list[dict], skip: str | None = None) -> dict:
    """cars: [{name, speed, enter_t}] in entry order. Returns the event log."""
    cars = [c for c in cars if c["name"] != skip]
    st = {c["name"]: {"x": 0.0, "mode": "waiting", "v": c["speed"]} for c in cars}
    events = {c["name"]: [] for c in cars}
    pullovers = []
    waiting_for = {}
    t = 0.0
    while any(s["mode"] != "done" for s in st.values()) and t < 120:
        lane = [n for n, s in st.items() if s["mode"] == "lane"]
        for c in cars:                                          # entries, in order
            s = st[c["name"]]
            if s["mode"] == "waiting" and t >= c["enter_t"] - 1e-9:
                if all(st[n]["x"] >= CAR + REJOIN_AHEAD for n in lane):
                    s["mode"] = "lane"
                    events[c["name"]].append(["enter", round(t, 3), 0.0])
                    lane.append(c["name"])
                break                                            # later cars queue behind the first waiting one
        for n in lane:
            st[n]["x"] += st[n]["v"] * DT
        order = sorted(lane, key=lambda n: st[n]["x"])
        for back, front in zip(order, order[1:]):
            if st[front]["mode"] != "lane":
                continue
            gap = st[front]["x"] - st[back]["x"] - CAR
            if gap < CATCH_GAP and st[back]["v"] > st[front]["v"]:
                st[front]["mode"] = "shoulder"
                waiting_for[front] = back
                events[front].append(["pullover", round(t + DT, 3), round(st[front]["x"], 4)])
                pullovers.append({"car": front, "passer": back, "t": round(t + DT, 3)})
        for n, s in st.items():
            if s["mode"] == "shoulder":
                x = s["x"]
                passer_ok = st[waiting_for[n]]["x"] > x + CAR + REJOIN_AHEAD or st[waiting_for[n]]["mode"] == "done"
                clear = all(not (x - REJOIN_BEHIND < st[m]["x"] < x + CAR + REJOIN_AHEAD)
                            for m in st if m != n and st[m]["mode"] == "lane")
                if passer_ok and clear:
                    s["mode"] = "lane"
                    events[n].append(["rejoin", round(t + DT, 3), round(x, 4)])
            elif s["mode"] == "lane" and s["x"] >= ROAD:
                s["mode"] = "done"
                events[n].append(["finish", round(t + DT, 3), ROAD])
        t += DT
    finish = {n: ev[-1][1] for n, ev in events.items() if ev and ev[-1][0] == "finish"}
    return {"events": events, "pullovers": pullovers, "finish": finish}


def position(events: list, speed: float, t: float) -> tuple[float, str]:
    """Rebuild (x, mode) at time t from a car's event log."""
    if not events or t < events[0][1]:
        return 0.0, "waiting"
    x, mode, t0 = 0.0, "lane", events[0][1]
    for kind, te, xe in events[1:]:
        if t < te:
            break
        x = xe if kind != "finish" else ROAD
        mode = {"pullover": "shoulder", "rejoin": "lane", "finish": "done"}[kind]
        t0 = te
    if mode == "lane":
        x = min(ROAD, x + speed * (t - t0))
    return x, mode


def build(rng: random.Random) -> dict:
    while True:
        n = rng.randint(4, 6)
        cars, t_in = [], 0.0
        for name, color in CARS[:n]:
            cars.append({"name": name, "color": color, "speed": round(rng.uniform(1.0, 2.2) / 0.05) * 0.05,
                         "enter_t": round(t_in, 2)})
            t_in += rng.uniform(0.4, 2.2)
        for c in cars:
            c["speed"] = round(c["speed"], 2)
        res = simulate(cars)
        if len(res["finish"]) != n:
            continue
        end = max(res["finish"].values())
        n_pull = len(res["pullovers"])
        pulled = {p["car"] for p in res["pullovers"]}
        if 2 <= n_pull <= 6 and end <= 20 and len(pulled) >= 2:
            fin = sorted(res["finish"].values())
            if all(b - a >= 0.25 for a, b in zip(fin, fin[1:])):
                return {"road": ROAD, "cars": cars, **res}


def check(scene: dict) -> list[str]:
    errs = []
    if simulate(scene["cars"]) != {k: scene[k] for k in ("events", "pullovers", "finish")}:
        errs.append("stored event log doesn't match a replay")
    speeds = {c["name"]: c["speed"] for c in scene["cars"]}
    for n, ev in scene["events"].items():                       # the log must be physically consistent
        for (k1, t1, x1), (k2, t2, x2) in zip(ev, ev[1:]):
            if k1 in ("enter", "rejoin"):
                if abs((x2 if k2 != "finish" else scene["road"]) - x1 - speeds[n] * (t2 - t1)) > 0.02:
                    errs.append(f"car {n} doesn't drive at its own speed between {k1} and {k2}")
            elif k1 == "pullover" and (k2 != "rejoin" or x2 != x1):
                errs.append(f"car {n} moves while on the shoulder")
    for k in range(0, 400):                                     # no overlapping cars in the lane
        t = k * 0.05
        lane = sorted(position(ev, speeds[n], t)[0] for n, ev in scene["events"].items()
                      if position(ev, speeds[n], t)[1] == "lane")
        if any(b - a < CAR - 0.02 for a, b in zip(lane, lane[1:])):
            errs.append(f"two cars overlap in the lane at t={t:.2f}")
            break
    return errs
