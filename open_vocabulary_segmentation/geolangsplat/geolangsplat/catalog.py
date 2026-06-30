# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Multi-prompt segment catalog: cluster every query's segments into a browsable,
ID'd table of objects you can pull out as ``.ply`` -- the notebook entry point.

A single ``segment`` answers "where is X". A *catalog* runs a whole vocabulary in
one pass and turns the result into a small object database. The flow per build:

    score the vocab (one SAM3 pass)  -->  threshold each prompt + split it into
    spatial objects (3D connected components)  -->  merge objects that several
    prompts land on (e.g. ``car`` and ``vehicle``) by 3D overlap, give each a
    stable id  -->  expose a table (pandas ``DataFrame``) + per-object ``.ply``.

Everything here is training-free and runs on CPU once the per-prompt scores
exist; only ``.ply`` extraction / rendering touches the splat tensors.

    cat = engine.catalog(["building", "car", "tree", "road"])
    cat.table                      # browse: id, label, n_gaussians, centroid, size
    cat.show()                     # top-down render with ids drawn on each object
    cat.extract(3, "obj3.ply")     # pull one object out as a .ply
    cat.export_all("scene_catalog/")  # a folder of per-object .ply + catalog.csv
"""
from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field

import numpy as np
import torch

from . import outputs as _out
from .instances import connected_components
from .select import select_query, spatial_cleanup

# A general starting vocabulary. Override it with the prompts that fit your scene
# (outdoor/aerial defaults shown); object/interior scenes want furniture words, etc.
DEFAULT_CATALOG_VOCAB: tuple[str, ...] = (
    "building",
    "house",
    "car",
    "road",
    "tree",
    "grass",
    "water",
    "sidewalk",
)

# Table column order (kept stable so a cudf/pandas swap is a one-liner).
COLUMNS: tuple[str, ...] = (
    "id",
    "label",
    "n_gaussians",
    "score",
    "cx",
    "cy",
    "cz",
    "dx",
    "dy",
    "dz",
    "prompts",
)


@dataclass
class CatalogObject:
    """One physical object in the catalog (a merged group of segment instances)."""

    id: int
    label: str  # best (highest-scoring) prompt for this object
    prompts: list[str]  # every prompt whose segment overlapped this object
    n_gaussians: int
    score: float
    centroid: np.ndarray  # (3,)
    bbox_min: np.ndarray  # (3,)
    bbox_max: np.ndarray  # (3,)

    @property
    def size(self) -> np.ndarray:
        """Axis-aligned bounding-box extent ``(dx, dy, dz)``."""
        return self.bbox_max - self.bbox_min

    def row(self) -> dict:
        """Flat dict for one table row (the unit a DataFrame is built from)."""
        c, s = self.centroid, self.size
        return {
            "id": self.id,
            "label": self.label,
            "n_gaussians": self.n_gaussians,
            "score": round(float(self.score), 4),
            "cx": round(float(c[0]), 4),
            "cy": round(float(c[1]), 4),
            "cz": round(float(c[2]), 4),
            "dx": round(float(s[0]), 4),
            "dy": round(float(s[1]), 4),
            "dz": round(float(s[2]), 4),
            "prompts": ", ".join(self.prompts),
        }


@dataclass
class SegmentCatalog:
    """A browsable, ID'd table of objects, ready to inspect / extract in a notebook.

    ``labels`` is a ``[N]`` long tensor in input ``.ply`` order: ``-1`` for an
    unassigned Gaussian, else the owning object's :attr:`CatalogObject.id`.
    """

    objects: list[CatalogObject]
    labels: torch.Tensor  # [N] long, -1 = unassigned, else object id (CPU)
    config: object
    _model: object = None
    metadata: dict = field(default_factory=dict)
    _engine: object = None  # warm engine, if any -- lets browse() re-query in place

    # -- container sugar ----------------------------------------------------

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, obj_id: int) -> CatalogObject:
        return self._by_id(obj_id)

    def __iter__(self):
        return iter(self.objects)

    def __repr__(self) -> str:
        head = f"SegmentCatalog: {len(self.objects)} objects from {len(self._vocab())} prompts"
        rows = [f"  #{o.id}: {o.label!r}  {o.n_gaussians:,} gaussians  score={o.score:.3f}" for o in self.objects[:20]]
        if len(self.objects) > 20:
            rows.append(f"  ... (+{len(self.objects) - 20} more)")
        return "\n".join([head, *rows])

    def _repr_html_(self):
        """Rich table when a catalog is the last expression in a notebook cell."""
        try:
            import pandas as pd  # noqa: PLC0415 - optional, notebook-only

            head = f"<b>SegmentCatalog</b>: {len(self.objects)} objects from {len(self._vocab())} prompts"
            return head + pd.DataFrame(self.rows(), columns=list(COLUMNS)).to_html(index=False)
        except ModuleNotFoundError:
            return None  # Jupyter falls back to __repr__

    def _vocab(self) -> list[str]:
        seen: list[str] = []
        for o in self.objects:
            for p in o.prompts:
                if p not in seen:
                    seen.append(p)
        return seen

    # -- table --------------------------------------------------------------

    def rows(self) -> list[dict]:
        """The catalog as a list of flat row dicts (backend-agnostic)."""
        return [o.row() for o in self.objects]

    @property
    def table(self):
        """The catalog as a DataFrame for notebook browsing.

        Uses pandas (swap to cudf for a GPU frame -- same columns). Falls back to
        the plain :meth:`rows` list if no DataFrame library is installed.
        """
        rows = self.rows()
        try:
            import pandas as pd  # noqa: PLC0415 - optional, notebook-only

            return pd.DataFrame(rows, columns=list(COLUMNS))
        except ModuleNotFoundError:
            return rows

    # -- masks / extraction -------------------------------------------------

    def mask(self, obj_id: int) -> torch.Tensor:
        """Boolean ``[N]`` mask selecting only object ``obj_id`` (model device)."""
        self._by_id(obj_id)
        m = self.labels == obj_id
        dev = getattr(getattr(self._model, "means", None), "device", None)
        return m.to(dev) if dev is not None else m

    @property
    def all_mask(self) -> torch.Tensor:
        """Boolean ``[N]`` mask of every catalogued Gaussian."""
        m = self.labels >= 0
        dev = getattr(getattr(self._model, "means", None), "device", None)
        return m.to(dev) if dev is not None else m

    def extract(self, obj_id: int, path) -> int:
        """Write only object ``obj_id`` to ``path`` as a ``.ply``. Returns its count."""
        return _out.write_ply_segmented(self._model, self.mask(obj_id), path, metadata=self.metadata)

    def export_all(self, out_dir, *, labeled_ply: bool = True) -> pathlib.Path:
        """Write the whole catalog to a folder -- the "segment database" to browse
        or hand to Omniverse:

        * ``catalog.csv``            -- the table (one row per object),
        * ``objects/<id>_<label>.ply`` -- each object as its own splat,
        * ``catalog_labeled.ply``    -- the full splat recoloured by object (optional).
        """
        out = pathlib.Path(out_dir)
        (out / "objects").mkdir(parents=True, exist_ok=True)
        (out / "catalog.csv").write_text(_to_csv(self.rows()))
        for o in self.objects:
            self.extract(o.id, out / "objects" / f"{o.id:03d}_{_slug(o.label)}.ply")
        if labeled_ply:
            _out.write_ply_labels(
                self._model,
                self.labels.to(self._model.means.device),
                out / "catalog_labeled.ply",
                metadata=self.metadata,
            )
        print(f"[catalog] wrote {len(self.objects)} objects -> {out}", flush=True)
        return out

    # -- persistence --------------------------------------------------------

    def save(self, out_dir) -> pathlib.Path:
        """Persist the catalog (table + per-Gaussian object labels) so it can be
        reloaded later with :meth:`load` and the same source ``.ply``."""
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "catalog.csv").write_text(_to_csv(self.rows()))
        np.savez_compressed(out / "labels.npz", labels=self.labels.cpu().numpy())
        meta = {"objects": self.rows()}
        (out / "catalog.json").write_text(json.dumps(meta, indent=2))
        print(f"[catalog] saved -> {out}", flush=True)
        return out

    @classmethod
    def load(cls, in_dir, model, config=None, *, metadata: dict | None = None) -> "SegmentCatalog":
        """Reload a catalog saved by :meth:`save`, bound to ``model`` for extraction."""
        in_dir = pathlib.Path(in_dir)
        labels = torch.from_numpy(np.load(in_dir / "labels.npz")["labels"]).long()
        meta = json.loads((in_dir / "catalog.json").read_text())
        objects = [_object_from_row(r) for r in meta["objects"]]
        return cls(objects=objects, labels=labels, config=config, _model=model, metadata=metadata or {})

    # -- visualization ------------------------------------------------------

    def show(self, *, size: int = 900, device=None, path=None):
        """Inline top-down render with each object's id drawn on it (returns a
        ``PIL.Image``; pass ``path`` to also save the PNG)."""
        from .instances import _annotate, _render_topdown

        pil, uv = _render_topdown(self._model, self.config, self.objects, size=size, device=device)
        labels = [(int(o.id), uv[i]) for i, o in enumerate(self.objects) if uv[i] is not None]
        out = _annotate(pil, labels)
        if path is not None:
            out.save(path)
            print(f"[catalog] wrote {path}", flush=True)
        return out

    def show_one(self, obj_id: int, *, size: int = 900, device=None, path=None):
        """Top-down render with only object ``obj_id`` highlighted (rest dimmed)."""
        from .instances import _annotate, _render_topdown

        self._by_id(obj_id)
        i = next(k for k, o in enumerate(self.objects) if o.id == obj_id)
        pil, uv = _render_topdown(
            self._model, self.config, self.objects, size=size, device=device, highlight=self.mask(obj_id)
        )
        out = _annotate(pil, [(obj_id, uv[i])] if uv[i] is not None else [])
        if path is not None:
            out.save(path)
            print(f"[catalog] wrote {path}", flush=True)
        return out

    def preview(self, ids=None, *, size: int = 700, device=None):
        """Top-down render with object ``ids`` highlighted (all dimmed if empty).

        ``ids`` may be a single id or an iterable; returns a ``PIL.Image``.
        """
        from .instances import _annotate, _render_topdown

        ids = [] if ids is None else ([ids] if isinstance(ids, int) else [int(i) for i in ids])
        highlight = self._ids_mask(ids).to(self._model.means.device) if ids else None
        pil, uv = _render_topdown(self._model, self.config, self.objects, size=size, device=device, highlight=highlight)
        keep = set(ids) if ids else {o.id for o in self.objects}
        labels = [(o.id, uv[i]) for i, o in enumerate(self.objects) if uv[i] is not None and o.id in keep]
        return _annotate(pil, labels)

    def browse(self, *, size: int = 600, out_dir: str = "picked", device=None):
        """Interactive notebook picker -- the no-id-juggling way to grab segments.

        Renders a live preview beside a multi-select list of every object: click
        objects to highlight them in the render, then hit **Export selected** to
        write their ``.ply`` files. If the catalog has a warm engine attached
        (``engine.catalog`` / ``build_catalog``), a query box re-runs the vocabulary
        in place. Requires ``ipywidgets`` in a Jupyter kernel.
        """
        try:
            import ipywidgets as widgets  # noqa: PLC0415 - optional, notebook-only
        except ModuleNotFoundError as exc:  # pragma: no cover - needs a notebook
            raise RuntimeError(
                "browse() needs ipywidgets (pip install ipywidgets); use .table / .extract otherwise"
            ) from exc
        import io  # noqa: PLC0415

        from IPython.display import display  # noqa: PLC0415 - notebook-only

        def _png(ids) -> bytes:
            buf = io.BytesIO()
            self.preview(ids, size=size, device=device).save(buf, format="PNG")
            return buf.getvalue()

        def _options():
            return [(f"#{o.id}  {o.label}  ({o.n_gaussians:,} gaussians)", o.id) for o in self.objects]

        picker = widgets.SelectMultiple(
            options=_options(), rows=min(16, max(4, len(self.objects))), layout=widgets.Layout(width="340px")
        )
        img = widgets.Image(value=_png([]), format="png")
        outdir = widgets.Text(value=out_dir, description="out dir:", layout=widgets.Layout(width="280px"))
        export = widgets.Button(description="Export selected", button_style="primary", icon="download")
        status = widgets.Output()

        def _on_select(_change):
            img.value = _png(list(picker.value))

        picker.observe(_on_select, names="value")

        def _on_export(_btn):
            with status:
                status.clear_output()
                ids = list(picker.value) or [o.id for o in self.objects]
                dest = pathlib.Path(outdir.value)
                (dest / "objects").mkdir(parents=True, exist_ok=True)
                for i in ids:
                    obj = self._by_id(i)
                    path = dest / "objects" / f"{i:03d}_{_slug(obj.label)}.ply"
                    self.extract(i, path)
                    print(f"  #{i}  {obj.label}  ->  {path}")
                print(f"exported {len(ids)} object(s) to {dest}/")

        export.on_click(_on_export)
        controls = [picker, widgets.HBox([outdir, export]), status]

        if getattr(self, "_engine", None) is not None:
            query = widgets.Text(
                placeholder="type a vocab and Search (comma/space separated)", layout=widgets.Layout(width="280px")
            )
            search = widgets.Button(description="Search", icon="search")

            def _on_search(_btn):
                with status:
                    status.clear_output()
                    words = [w for w in re.split(r"[,\s]+", query.value) if w]
                    if not words:
                        print("type a vocab first")
                        return
                    print(f"re-querying {words} ...")
                    fresh = self._engine.catalog(words)
                    self.objects, self.labels, self.metadata = fresh.objects, fresh.labels, fresh.metadata
                    picker.options = _options()
                    picker.value = ()
                    img.value = _png([])
                    print(f"{len(self.objects)} objects")

            search.on_click(_on_search)
            controls = [widgets.HBox([query, search]), *controls]

        ui = widgets.HBox([widgets.VBox(controls), img])
        display(ui)
        return ui

    # -- internals ----------------------------------------------------------

    def _ids_mask(self, ids) -> torch.Tensor:
        """Boolean ``[N]`` mask (CPU) selecting the union of object ``ids``."""
        m = torch.zeros_like(self.labels, dtype=torch.bool)
        for i in ids:
            m |= self.labels == int(i)
        return m

    def _by_id(self, obj_id: int) -> CatalogObject:
        for o in self.objects:
            if o.id == obj_id:
                return o
        valid = f"0..{len(self.objects) - 1}" if self.objects else "(empty catalog)"
        raise KeyError(f"no object with id {obj_id}; valid ids: {valid}")


# --- clustering core (pure, CPU-testable) ----------------------------------


def _pair_intersections(covers: list[torch.Tensor], n_obj: int) -> dict[tuple[int, int], int]:
    """Gaussian-count of the intersection of every pair of objects from *different*
    prompts. ``covers[c]`` is ``[N]`` long: the global object id covering each
    Gaussian for prompt ``c`` (``-1`` = none). Objects from the same prompt are
    disjoint (connected components), so only cross-prompt pairs can intersect.
    """
    inter: dict[tuple[int, int], int] = {}
    c = len(covers)
    for a in range(c):
        ca = covers[a]
        for b in range(a + 1, c):
            cb = covers[b]
            both = (ca >= 0) & (cb >= 0)
            if not bool(both.any()):
                continue
            key = ca[both].long() * n_obj + cb[both].long()
            uk, cnt = torch.unique(key, return_counts=True)
            for k_, cnt_ in zip(uk.tolist(), cnt.tolist()):
                i, j = divmod(int(k_), n_obj)
                pair = (i, j) if i < j else (j, i)
                inter[pair] = inter.get(pair, 0) + int(cnt_)
    return inter


def _union_find(n: int, edges) -> list[int]:
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return [find(i) for i in range(n)]


# --- assembly --------------------------------------------------------------


def catalog_from_scores(
    model,
    vocab,
    cfg,
    scores: torch.Tensor,
    *,
    seen: torch.Tensor,
    denom: torch.Tensor | None = None,
    select: float | None = None,
    iou: float | None = None,
    link_frac: float | None = None,
    min_size: int | None = None,
    metadata: dict | None = None,
) -> SegmentCatalog:
    """Assemble a :class:`SegmentCatalog` from per-prompt scores ``[N, C]``.

    For each prompt: threshold (reusing :func:`select_query`) -> spatial cleanup ->
    connected components. Then merge objects across prompts whose 3D IoU clears
    ``iou`` (``cfg.cat_iou``) and assign ids largest-first.
    """
    vocab = list(vocab)
    iou_thr = cfg.cat_iou if iou is None else iou
    means_cpu = model.means.detach().float().cpu()
    span = _robust_span(means_cpu)  # ignore far floaters so the voxel size tracks the real scene
    scores_cpu = scores.detach().float().cpu()
    if scores_cpu.ndim == 1:
        scores_cpu = scores_cpu.unsqueeze(1)
    seen_cpu = seen.detach().cpu() if seen is not None else torch.ones(means_cpu.shape[0])
    denom_cpu = denom.detach().cpu() if denom is not None else None
    n = means_cpu.shape[0]

    # 1. per-prompt selections -> spatial instances ("raw" objects, one per (prompt, component)).
    raw_masks: list[torch.Tensor] = []  # bool [N] (CPU)
    raw_label: list[str] = []
    raw_score: list[float] = []
    covers: list[torch.Tensor] = []  # per prompt: [N] long global-raw-id or -1
    for c, prompt in enumerate(vocab):
        sel = select_query(scores_cpu[:, c], seen_cpu, cfg, query=prompt, select=select, denom=denom_cpu, compete=False)
        sel = spatial_cleanup(means_cpu, sel, cfg)
        comp_ids, infos = connected_components(means_cpu, sel, cfg, link_frac=link_frac, min_size=min_size, span=span)
        cover = torch.full((n,), -1, dtype=torch.long)
        for info in infos:
            local = info["idx"]
            m = comp_ids == local
            gid = len(raw_masks)
            cover[m] = gid
            raw_masks.append(m)
            raw_label.append(prompt)
            raw_score.append(float(scores_cpu[:, c][m].mean()) if bool(m.any()) else 0.0)
        covers.append(cover)

    if not raw_masks:
        return SegmentCatalog([], torch.full((n,), -1, dtype=torch.long), cfg, _model=model, metadata=metadata or {})

    # 2. merge raw objects that overlap across prompts (same physical thing).
    sizes = [int(m.sum()) for m in raw_masks]
    inter = _pair_intersections(covers, len(raw_masks))
    edges = []
    for (i, j), nij in inter.items():
        union = sizes[i] + sizes[j] - nij
        if union > 0 and (nij / union) >= iou_thr:
            edges.append((i, j))
    roots = _union_find(len(raw_masks), edges)

    groups: dict[int, list[int]] = {}
    for k, r in enumerate(roots):
        groups.setdefault(r, []).append(k)

    # 3. build one CatalogObject per group; assign ids largest-first; paint labels.
    built: list[tuple[torch.Tensor, str, list[str], float]] = []
    for members in groups.values():
        gmask = raw_masks[members[0]].clone()
        for k in members[1:]:
            gmask |= raw_masks[k]
        best = max(members, key=lambda k: raw_score[k])
        prompts = sorted({raw_label[k] for k in members})
        built.append((gmask, raw_label[best], prompts, raw_score[best]))

    built.sort(key=lambda t: int(t[0].sum()), reverse=True)
    labels = torch.full((n,), -1, dtype=torch.long)
    objects: list[CatalogObject] = []
    for obj_id, (gmask, label, prompts, score) in enumerate(built):
        labels[gmask] = obj_id  # later (smaller) objects can only overwrite empty slots below
        pts = means_cpu[gmask]
        objects.append(
            CatalogObject(
                id=obj_id,
                label=label,
                prompts=prompts,
                n_gaussians=int(gmask.sum()),
                score=score,
                centroid=pts.mean(dim=0).numpy(),
                bbox_min=pts.min(dim=0).values.numpy(),
                bbox_max=pts.max(dim=0).values.numpy(),
            )
        )
    # paint largest-first so overlapping smaller objects don't steal a bigger one's Gaussians
    labels.fill_(-1)
    for obj_id, (gmask, *_rest) in enumerate(built):
        labels[gmask & (labels < 0)] = obj_id

    return SegmentCatalog(objects=objects, labels=labels, config=cfg, _model=model, metadata=metadata or {})


# --- small helpers ---------------------------------------------------------


def _robust_span(means_cpu: torch.Tensor) -> float:
    """Scene extent from a 1-99% quantile box (per axis), ignoring far floaters.

    The full min/max range is dominated by stray Gaussians on large captures, which
    inflates the connected-components voxel and merges separate objects; the
    quantile box tracks the real scene. Subsampled for speed on huge splats.
    """
    mc = means_cpu
    if mc.shape[0] > 500_000:
        mc = mc[torch.randperm(mc.shape[0])[:500_000]]
    lo = torch.quantile(mc, 0.01, dim=0)
    hi = torch.quantile(mc, 0.99, dim=0)
    return float((hi - lo).max())


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s or "object"


def _to_csv(rows: list[dict]) -> str:
    header = ",".join(COLUMNS)
    lines = [header]
    for r in rows:
        vals = []
        for col in COLUMNS:
            v = r.get(col, "")
            v = f'"{v}"' if isinstance(v, str) and ("," in v) else v
            vals.append(str(v))
        lines.append(",".join(vals))
    return "\n".join(lines) + "\n"


def _object_from_row(r: dict) -> CatalogObject:
    prompts = [p.strip() for p in str(r.get("prompts", "")).split(",") if p.strip()]
    return CatalogObject(
        id=int(r["id"]),
        label=str(r["label"]),
        prompts=prompts,
        n_gaussians=int(r["n_gaussians"]),
        score=float(r["score"]),
        centroid=np.array([r["cx"], r["cy"], r["cz"]], dtype=np.float32),
        bbox_min=np.array([r["cx"] - r["dx"] / 2, r["cy"] - r["dy"] / 2, r["cz"] - r["dz"] / 2], dtype=np.float32),
        bbox_max=np.array([r["cx"] + r["dx"] / 2, r["cy"] + r["dy"] / 2, r["cz"] + r["dz"] / 2], dtype=np.float32),
    )
