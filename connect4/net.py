"""Small policy/value network (MLP) written in NumPy with manual backprop and Adam."""
import json
import base64
import numpy as np

INPUT = 84
N_ACTIONS = 7


class Net:
    def __init__(self, hidden=(384, 384, 256), value_hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        self.hidden = tuple(hidden)
        self.value_hidden = value_hidden
        self.params = {}
        sizes = (INPUT,) + self.hidden
        for i in range(len(self.hidden)):
            self.params[f"W{i}"] = he(rng, sizes[i], sizes[i + 1])
            self.params[f"b{i}"] = np.zeros(sizes[i + 1], np.float32)
        last = self.hidden[-1]
        self.params["Wp"] = he(rng, last, N_ACTIONS) * 0.1
        self.params["bp"] = np.zeros(N_ACTIONS, np.float32)
        self.params["Wv1"] = he(rng, last, value_hidden)
        self.params["bv1"] = np.zeros(value_hidden, np.float32)
        self.params["Wv2"] = he(rng, value_hidden, 1) * 0.1
        self.params["bv2"] = np.zeros(1, np.float32)
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    # ---------- inference ----------
    def forward(self, x):
        p = self.params
        hs = [x]
        h = x
        for i in range(len(self.hidden)):
            h = np.maximum(h @ p[f"W{i}"] + p[f"b{i}"], 0)
            hs.append(h)
        logits = h @ p["Wp"] + p["bp"]
        g = np.maximum(h @ p["Wv1"] + p["bv1"], 0)
        value = np.tanh(g @ p["Wv2"] + p["bv2"])[:, 0]
        return logits, value, (hs, g)

    def predict(self, x):
        logits, value, _ = self.forward(x)
        return softmax(logits), value

    # ---------- training ----------
    def train_step(self, x, pi, z, lr=1e-3, wd=1e-4):
        p = self.params
        B = x.shape[0]
        logits, value, (hs, g) = self.forward(x)
        probs = softmax(logits)
        policy_loss = -np.mean(np.sum(pi * np.log(probs + 1e-8), axis=1))
        value_loss = np.mean((value - z) ** 2)

        grads = {}
        d_logits = (probs - pi) / B
        h = hs[-1]
        grads["Wp"] = h.T @ d_logits
        grads["bp"] = d_logits.sum(0)
        dh = d_logits @ p["Wp"].T

        d_vpre = (2.0 * (value - z) / B * (1 - value ** 2))[:, None].astype(np.float32)
        grads["Wv2"] = g.T @ d_vpre
        grads["bv2"] = d_vpre.sum(0)
        dg = (d_vpre @ p["Wv2"].T) * (g > 0)
        grads["Wv1"] = h.T @ dg
        grads["bv1"] = dg.sum(0)
        dh += dg @ p["Wv1"].T

        for i in reversed(range(len(self.hidden))):
            da = dh * (hs[i + 1] > 0)
            grads[f"W{i}"] = hs[i].T @ da
            grads[f"b{i}"] = da.sum(0)
            if i > 0:
                dh = da @ p[f"W{i}"].T

        self.adam(grads, lr, wd)
        return float(policy_loss), float(value_loss)

    def adam(self, grads, lr, wd, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        c1 = 1 - b1 ** self.t
        c2 = 1 - b2 ** self.t
        for k, gk in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            step = lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + eps)
            if k.startswith("W"):
                step += lr * wd * self.params[k]
            self.params[k] -= step.astype(np.float32)

    # ---------- persistence ----------
    def get_weights(self):
        return {k: v.copy() for k, v in self.params.items()}

    def set_weights(self, w):
        self.params = {k: np.asarray(v, np.float32).copy() for k, v in w.items()}

    def save(self, path):
        np.savez(path, hidden=np.array(self.hidden), value_hidden=self.value_hidden,
                 t=self.t, **{f"p_{k}": v for k, v in self.params.items()},
                 **{f"m_{k}": v for k, v in self.m.items()},
                 **{f"v_{k}": v for k, v in self.v.items()})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(hidden=tuple(int(h) for h in d["hidden"]), value_hidden=int(d["value_hidden"]))
        for k in net.params:
            net.params[k] = d[f"p_{k}"]
            if f"m_{k}" in d:
                net.m[k] = d[f"m_{k}"]
                net.v[k] = d[f"v_{k}"]
        net.t = int(d["t"])
        return net

    def export_js(self, path, meta=None):
        """Write weights as a JS file (base64 float32) for the browser game."""
        layers = []
        for i in range(len(self.hidden)):
            layers.append(self._pack(f"W{i}", f"b{i}"))
        data = {
            "hidden": layers,
            "policy": self._pack("Wp", "bp"),
            "value1": self._pack("Wv1", "bv1"),
            "value2": self._pack("Wv2", "bv2"),
            "meta": meta or {},
        }
        with open(path, "w") as f:
            f.write("window.C4_WEIGHTS = " + json.dumps(data) + ";\n")

    def _pack(self, wk, bk):
        W = self.params[wk].astype("<f4")
        b = self.params[bk].astype("<f4")
        return {"in": int(W.shape[0]), "out": int(W.shape[1]),
                "W": base64.b64encode(W.tobytes()).decode(),
                "b": base64.b64encode(b.tobytes()).decode()}


def he(rng, n_in, n_out):
    return (rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
