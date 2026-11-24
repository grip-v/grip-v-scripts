"""Independent answers for shell_game_tracking, read from what the renderer draws.

The decoder sees only `render.track(scene)`: on every frame the cups' bottom
centres as an UNORDERED list, the tokens visible on that frame, and the
mats. It never reads the moves, the peeks or sim.play. It:

  * follows every cup from frame to frame (constant-velocity prediction,
    then the closest match; it gives up if a match is ever unclear), which
    is what a viewer has to do since the cups look alike;
  * reads where each cup stands (on a mat, possibly lifted straight up) or
    that it is moving; a move is a run of frames in which some cup is off
    its mat, and gives which mat each cup went to;
  * reads which cup hides which token from the tokens seen under lifted
    cups at the start, and checks every later peek against its tracking.

What-ifs are re-played on its own list of moves (as permutations of mats).
"""
from __future__ import annotations
import functools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import render  # noqa: E402
from sim import ORD, clock_text, tenths, frames_showing  # noqa: E402  (text formatting only)

TOL = 0.01
WIN = 7


class Model:
    def __init__(self, tr):
        self.tr = tr
        self.n_frames = tr["n_frames"]
        self.slots = np.array(tr["slots"], float)
        self.ns = len(self.slots)
        self.ok = True
        self._follow()
        self._read()
        self._tokens()

    def _follow(self):
        obs = [np.array(c, float) for c in self.tr["cups"]]
        n = len(obs[0])
        T = np.zeros((self.n_frames, n, 2))
        T[0] = obs[0]
        for f in range(1, self.n_frames):
            pred = T[f - 1] + (T[f - 1] - T[f - 2] if f >= 2 else 0)
            D = np.hypot(pred[:, None, 0] - obs[f][None, :, 0], pred[:, None, 1] - obs[f][None, :, 1])
            used_i, used_j = set(), set()
            pairs = sorted((D[i, j], i, j) for i in range(n) for j in range(n))
            for d, i, j in pairs:
                if i in used_i or j in used_j:
                    continue
                # the match must be clearly better than this cup's next best
                alt = min(D[i, k] for k in range(n) if k != j)
                if d > 2 and alt < 2 * d:
                    self.ok = False
                T[f, i] = obs[f][j]
                used_i.add(i)
                used_j.add(j)
        self.T = T
        self.n = n

    def _read(self):
        """at[f][cup] = mat index (resting, or lifted straight up from it) or -1 (moving); lifted[f][cup].
        A stretch of frames off the mat is a lift if the cup only goes up and comes back down onto the same
        mat (x never changes); otherwise it is part of a move."""
        T, S = self.T, self.slots
        at = np.full((self.n_frames, self.n), -1, dtype=int)
        lifted = np.zeros((self.n_frames, self.n), dtype=bool)
        for c in range(self.n):
            rest = []
            for f in range(self.n_frames):
                x, y = T[f, c]
                k = int(np.argmin(np.abs(S[:, 0] - x) + np.abs(S[:, 1] - y)))
                rest.append(k if abs(S[k, 0] - x) < TOL and abs(S[k, 1] - y) < TOL else -1)
            f = 0
            while f < self.n_frames:
                if rest[f] >= 0:
                    at[f, c] = rest[f]
                    f += 1
                    continue
                g = f
                while g + 1 < self.n_frames and rest[g + 1] < 0:
                    g += 1
                before = rest[f - 1] if f > 0 else None
                after = rest[g + 1] if g + 1 < self.n_frames else None
                home = before if before is not None else after
                if home is not None and (before is None or after is None or before == after) and \
                        all(abs(T[h, c, 0] - S[home, 0]) < TOL and T[h, c, 1] < S[home, 1] for h in range(f, g + 1)):
                    at[f:g + 1, c] = home
                    lifted[f:g + 1, c] = True
                f = g + 1
        self.at, self.lifted = at, lifted
        moving = (at < 0).any(axis=1)
        self.moves = []
        f = 0
        while f < self.n_frames:
            if moving[f]:
                g = f
                while g + 1 < self.n_frames and moving[g + 1]:
                    g += 1
                if f == 0 or g == self.n_frames - 1:
                    self.ok = False
                    break
                pairs = sorted((int(at[f - 1, c]), int(at[g + 1, c])) for c in range(self.n)
                               if at[f - 1, c] != at[g + 1, c])
                self.moves.append({"start": f, "end": g, "pairs": pairs,
                                   "cups": sorted(c for c in range(self.n) if at[f - 1, c] != at[g + 1, c])})
                f = g + 1
            else:
                f += 1
        # lifts after the start: single-cup peeks
        self.peeks = []
        lf = lifted.any(axis=1)
        start_end = 0
        while start_end < self.n_frames and lf[start_end]:
            start_end += 1
        self.reveal_end = start_end
        f = start_end
        while f < self.n_frames:
            if lf[f]:
                g = f
                while g + 1 < self.n_frames and lf[g + 1]:
                    g += 1
                cups = sorted(set(int(c) for h in range(f, g + 1) for c in np.nonzero(lifted[h])[0]))
                if len(cups) != 1:
                    self.ok = False
                self.peeks.append({"start": f, "end": g, "cup": cups[0], "slot": int(at[f, cups[0]]),
                                   "tokens": self._seen(f, g)})
                f = g + 1
            else:
                f += 1

    def _seen(self, f0, f1):
        """{token key: cup} for tokens drawn in frames f0..f1, matched to the lifted cup above them."""
        out = {}
        for f in range(f0, f1 + 1):
            for t in self.tr["tokens"][f]:
                key = (t["colour"], t["shape"])
                cs = [c for c in range(self.n) if self.lifted[f, c] and abs(self.T[f, c, 0] - t["x"]) < TOL
                      and self.T[f, c, 1] < t["y"]]
                if not cs:
                    self.ok = False
                    continue
                cs = [min(cs, key=lambda c: t["y"] - self.T[f, c, 1])]       # the lifted cup right above it
                if out.get(key, cs[0]) != cs[0]:
                    self.ok = False
                out[key] = cs[0]
        return out

    def _tokens(self):
        self.tok_cup = self._seen(0, self.reveal_end)
        self.tok_keys = sorted(self.tok_cup, key=lambda k: self.at[0, self.tok_cup[k]])
        for p in self.peeks:
            for key, c in self.tok_cup.items():
                seen = key in p["tokens"]
                if seen != (c == p["cup"]):
                    self.ok = False

    # -------------------------------------------------------------- helpers
    def slot_after(self, cup, k):
        """Mat of a cup after the first k moves."""
        if k == 0:
            return int(self.at[0, cup])
        return int(self.at[self.moves[k - 1]["end"] + 1, cup])

    def tok(self, name):
        for key in self.tok_cup:
            if f"the {key[0]} {key[1]}" == name:
                return self.tok_cup[key]
        return None

    def rows(self):
        ys = sorted(set(round(float(y), 3) for y in self.slots[:, 1]))
        return [sorted([s for s in range(self.ns) if round(float(self.slots[s, 1]), 3) == y],
                       key=lambda s: self.slots[s, 0]) for y in ys]

    def name(self, s):
        rows = self.rows()
        if len(rows) == 1:
            r = rows[0]
            k = r.index(s)
            if k == 0:
                return "the leftmost position"
            if k == len(r) - 1:
                return "the rightmost position"
            if len(r) == 3:
                return "the middle position"
            return f"the {ORD[k]} position from the left"
        for ri, r in enumerate(rows):
            if s in r:
                col = ["left", "right"] if len(r) == 2 else ["left", "middle", "right"]
                return f"the {'top' if ri == 0 else 'bottom'}-{col[r.index(s)]} position"

    def by_name(self, nm):
        for s in range(self.ns):
            if self.name(s) == nm:
                return s
        return None

    def perm_run(self, moves, start_slot):
        s = start_slot
        for m in moves:
            for a, b in m:
                if a == s:
                    s = b
                    break
        return s


@functools.lru_cache(maxsize=16)
def _model_key(key: str) -> Model:
    return Model(render.track(json.loads(key)))


def _model(scene) -> Model:
    return _model_key(json.dumps(scene, sort_keys=True))


def answer(key: str, p: dict, scene: dict):
    return _answer(key, p, _model(scene))


def _answer(key, p, M: Model):
    if not M.ok:
        return None
    c = M.tok(p["tok"]) if "tok" in p else None
    nm = len(M.moves)
    pairs = [m["pairs"] for m in M.moves]
    if key == "n_cups":
        return M.n
    if key == "n_moves":
        return nm
    if key == "n_tokens":
        return len(M.tok_cup)
    if key == "start_slot":
        return M.name(M.slot_after(c, 0))
    if key == "first_swap":
        a, b = sorted({x for pr in pairs[0] for x in pr}, key=lambda s: (M.slots[s][1], M.slots[s][0]))
        return f"{M.name(a)} and {M.name(b)}"
    if key == "n_peeks":
        return len(M.peeks)
    if key == "peek_has":
        pk = M.peeks[p["p"]]
        return "yes" if pk["tokens"] else "no"
    if key == "peek_slot":
        return M.name(M.peeks[p["p"]]["slot"])
    if key == "in_move":
        k = p["k"]
        return "yes" if c in M.moves[k]["cups"] else "no"
    if key == "end_side":
        a, b = M.slots[M.slot_after(c, 0)][0], M.slots[M.slot_after(c, nm)][0]
        return "further left" if b < a else "further right" if b > a else "in the same position"
    if key == "first_mover":
        ca, cb = M.tok(p["a"]), M.tok(p["b"])
        fa = next((m["start"] for m in M.moves if ca in m["cups"]), None)
        fb = next((m["start"] for m in M.moves if cb in m["cups"]), None)
        if fa is None or fb is None or fa == fb:
            return None
        return p["a"] + "'s cup" if fa < fb else p["b"] + "'s cup"
    if key == "left_end":
        ca, cb = M.tok(p["a"]), M.tok(p["b"])
        xa, xb = M.slots[M.slot_after(ca, nm)][0], M.slots[M.slot_after(cb, nm)][0]
        return p["a"] if xa < xb else p["b"]
    if key == "first_dest":
        k = next(i for i, m in enumerate(M.moves) if c in m["cups"])
        return M.name(M.slot_after(c, k + 1))
    if key == "moved_more":
        ca = next(x for x in range(M.n) if M.slot_after(x, 0) == M.by_name(p["a"]))
        cb = next(x for x in range(M.n) if M.slot_after(x, 0) == M.by_name(p["b"]))
        na, nb = (sum(1 for m in M.moves if x in m["cups"]) for x in (ca, cb))
        if na == nb:
            return None
        return f"the cup that starts in {p['a'] if na > nb else p['b']}"
    if key == "leave_first":
        sa, sb = M.by_name(p["a"]), M.by_name(p["b"])
        fa = next(m["start"] for m in M.moves if any(x == sa for x, _ in m["pairs"]))
        fb = next(m["start"] for m in M.moves if any(x == sb for x, _ in m["pairs"]))
        return None if fa == fb else (p["a"] if fa < fb else p["b"])
    if key == "tok_moves":
        return sum(1 for m in M.moves if c in m["cups"])
    if key == "slot_moves":
        s = M.by_name(p["slot"])
        return sum(1 for m in M.moves if any(a == s for a, _ in m["pairs"]))
    if key == "first_move_time":
        return clock_text(tenths(next(m["start"] for m in M.moves if c in m["cups"])))
    if key == "move_time":
        return clock_text(tenths(M.moves[p["k"]]["start"]))
    if key == "moves_between":
        f1, f2 = frames_showing(p["t1"])[1], frames_showing(p["t2"])[0]
        return sum(1 for m in M.moves if f1 < m["start"] < f2)
    if key == "tok_visits":
        return len({M.slot_after(c, k) for k in range(nm + 1)})
    if key == "final_slot":
        return M.name(M.slot_after(c, nm))
    if key == "slot_at":
        a, b = frames_showing(p["t"])
        vals = {int(M.at[f, c]) for f in range(a - WIN, b + WIN + 1)}
        return M.name(vals.pop()) if len(vals) == 1 and -1 not in vals else None
    if key == "returned":
        home = M.slot_after(c, 0)
        away = False
        for k in range(1, nm):
            s = M.slot_after(c, k)
            if s != home:
                away = True
            elif away:
                return "yes"
        return "no"
    if key == "cup_end":
        s = M.by_name(p["slot"])
        cup = next(x for x in range(M.n) if M.slot_after(x, 0) == s)
        return M.name(M.slot_after(cup, nm))
    if key == "token_at_end":
        s = M.by_name(p["slot"])
        hit = [k for k, cup in M.tok_cup.items() if M.slot_after(cup, nm) == s]
        return f"the {hit[0][0]} {hit[0][1]}" if hit else "no token"
    if key == "peek_origin":
        return M.name(M.slot_after(M.peeks[p["p"]]["cup"], 0))
    # what-ifs: replay the drawn moves as permutations of mats
    start = M.slot_after(c, 0) if c is not None else None
    if key == "cf_skip":
        return M.name(M.perm_run([pr for i, pr in enumerate(pairs) if i != p["k"]], start))
    if key == "cf_extra":
        x, y = M.by_name(p["x"]), M.by_name(p["y"])
        return M.name(M.perm_run(pairs + [[(x, y), (y, x)]], start))
    if key == "cf_start":
        return M.name(M.perm_run(pairs, M.by_name(p["slot"])))
    if key == "cf_order":
        k = p["k"]
        pr = list(pairs)
        pr[k], pr[k + 1] = pr[k + 1], pr[k]
        return M.name(M.perm_run(pr, start))
    return None
