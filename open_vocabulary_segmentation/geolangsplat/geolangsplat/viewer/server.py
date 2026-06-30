# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Interactive web catalog browser for GeoLangSplat (backs ``gls explore``).

A small web UI drives the object catalog and the fvdb viewer renders the result.
The flow is one word at a time:

1. type a class ("building", "car") and Search,
2. every instance of it lights up in its own colour in the 3D view,
3. click an instance in the list -- the viewer swaps to a **cutout** of just that
   object (the rest of the scene is removed),
4. Back returns to the highlighted view so you can pick another or search again.

An optional visual companion for eyeballing and curation; the stable surface is the
``segment`` / ``build_catalog`` APIs and the ``gls`` CLI.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import torch

from ..cameras import orbit_cameras
from ..config import GeoLangSplatConfig, apply_recipe
from ..engine import GeoLangSplatEngine
from ..outputs import SH_C0


def _frame_viewer(scene, engine, cfg) -> None:
    """Orient + frame the viewer camera for the scene.

    Sets the up axis (object/COLMAP plys aren't z-up), shrinks the near clip plane
    (the auto default is too far to zoom *into* small object scenes), and starts the
    camera at a close, low vantage. Note: fvdb.viz's orbit control clamps how far
    *below* the orbit plane you can drag (a ground-plane assumption), so the
    rendered orbit -- which includes below-horizon views for object captures -- is
    the source of truth; the viewer is just for eyeballing.
    """
    import numpy as np

    from .. import autoview

    up = autoview.resolve_up(getattr(cfg, "up", "auto"), engine.means)
    try:
        m = engine.means.detach().float().cpu().numpy()
        lo = np.quantile(m, 0.02, axis=0)
        hi = np.quantile(m, 0.98, axis=0)
        center = (lo + hi) / 2
        radius = max(float(np.linalg.norm(hi - lo)) / 2, 0.001)
        scene.camera_up_direction = up
        scene.camera_near = max(radius * 0.002, 0.0001)
        scene.camera_far = radius * 100
        c2w = orbit_cameras(center, 18, radius * 2.2, num_azimuth=1, azimuth_offset_deg=35, up=up)[0][1]
        scene.set_camera_lookat(c2w[:3, 3], center, up)
    except Exception as e:
        print(f"[viewer] could not frame scene (up={getattr(cfg, 'up', '+z')!r}): {e}", flush=True)


_CHIPS = ("building", "car", "tree", "road")


def _object_palette(k: int) -> list[tuple[float, float, float]]:
    """``k`` visually distinct RGB colors (golden-ratio hue walk) for object recolor."""
    import colorsys

    return [colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.65, 1.0) for i in range(max(k, 1))]


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>GeoLangSplat catalog</title>
<style>
body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0c0e12;color:#e8eaed}
.bar{padding:10px 16px;background:#13161c;border-bottom:1px solid #222;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
input#q{flex:1;min-width:200px;padding:10px 12px;font-size:16px;border-radius:8px;border:1px solid #2a2f3a;background:#0c0e12;color:#fff}
button{padding:9px 14px;font-size:14px;border-radius:8px;border:1px solid #2a2f3a;background:#1d2330;color:#e8eaed;cursor:pointer}
button:hover{background:#2a3344}
button.primary{background:#1f6feb;border-color:#1f6feb;color:#fff}
.chip{padding:6px 10px;border-radius:999px;background:#1a1f29;border:1px solid #2a2f3a;font-size:13px}
.knob{display:flex;gap:6px;align-items:center;font-size:12px;color:#9aa0aa}
#status{padding:6px 16px;font-size:13px;color:#7fd18b;background:#0f1218}
.main{display:flex;height:calc(100vh - 112px)}
#panel{width:300px;min-width:300px;overflow:auto;background:#0f1218;border-right:1px solid #222}
#panel .row{padding:9px 14px;border-bottom:1px solid #181c24;cursor:pointer;display:flex;justify-content:space-between;gap:8px}
#panel .row:hover{background:#1d2330}
#panel .row .n{color:#7a808a;font-size:12px}
#backbar{padding:9px 14px;border-bottom:1px solid #222;display:none}
iframe{flex:1;border:0;background:#000}
</style></head><body>
<div class=bar>
 <input id=q placeholder="One class at a time, e.g. building, car" autofocus>
 <button class=primary onclick=search()>Search</button>
 <span id=chips></span>
 <span class=knob>conf <input id=select type=range min=0 max=1 step=0.01 value=__SELECT__ oninput="sv.textContent=this.value"><span id=sv>__SELECT__</span></span>
 <span class=knob title="voxel link size as a fraction of scene span: smaller separates closer/smaller objects, larger merges them">split <input id=gran type=range min=0.002 max=0.03 step=0.001 value=0.008 oninput="gv.textContent=this.value"><span id=gv>0.008</span></span>
 <span class=knob>min <input id=minsize type=number value=50 min=1 style="width:64px;padding:6px;border-radius:6px;border:1px solid #2a2f3a;background:#0c0e12;color:#fff"></span>
 <a id=viewlink target=_blank class=chip style="text-decoration:none">Open 3D viewer &#8599;</a>
</div>
<div id=status>type a class and Search &mdash; each instance lights up; click one to cut it out</div>
<div class=main>
 <div id=panel>
  <div id=backbar><button onclick=back()>&#8592; Back to all</button> <button onclick=exportall()>Export all</button></div>
  <div id=list></div>
 </div>
 <iframe id=view></iframe>
</div>
<script>
const DEFAULT_CHIPS=__CHIPS__;
const cc=document.getElementById('chips');
DEFAULT_CHIPS.forEach(c=>{const b=document.createElement('button');b.className='chip';b.textContent=c;b.onclick=()=>{document.getElementById('q').value=c;search()};cc.appendChild(b)});
const vurl='http://'+location.hostname+':__VIEWPORT__';
document.getElementById('view').src=vurl;
document.getElementById('viewlink').href=vurl;
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')search()});
function st(t){document.getElementById('status').textContent=t}
function search(){
 const q=document.getElementById('q').value.trim();if(!q)return;
 const s=document.getElementById('select').value,g=document.getElementById('gran').value,m=document.getElementById('minsize').value;
 st('finding "'+q+'" instances ...');
 fetch('/catalog?q='+encodeURIComponent(q)+'&select='+s+'&gran='+g+'&min='+m)
  .then(r=>r.json()).then(d=>{
   if(d.error){st('error: '+d.error);return;}
   renderlist(d.word,d.rows);
   st('"'+d.word+'"  ->  '+d.n+' instance(s) highlighted; click one to cut it out');
  }).catch(e=>st('error: '+e));
}
function renderlist(word,rows){
 document.getElementById('backbar').style.display='none';
 const L=document.getElementById('list');L.innerHTML='';
 rows.forEach(r=>{
  const d=document.createElement('div');d.className='row';
  d.innerHTML='<span>'+word+' #'+r.id+'</span><span class=n>'+r.n.toLocaleString()+'</span>';
  d.onclick=()=>cutout(r.id,word);L.appendChild(d);
 });
 if(!rows.length){L.innerHTML='<div class=row><span class=n>no instances &mdash; lower conf or split</span></div>';}
}
function cutout(id,word){
 st('cutting out '+word+' #'+id+' ...');
 fetch('/cutout?id='+id).then(r=>r.json()).then(d=>{
  if(d.error){st('error: '+d.error);return;}
  document.getElementById('backbar').style.display='block';
  st(word+' #'+id+'  ->  '+(d.n||0).toLocaleString()+' gaussians (cutout; rest hidden)');
 }).catch(e=>st('error: '+e));
}
function back(){fetch('/back').then(r=>r.json()).then(()=>{document.getElementById('backbar').style.display='none';st('back to all instances')});}
function exportall(){st('exporting ...');fetch('/export').then(r=>r.json()).then(d=>{if(d.error){st('error: '+d.error);return;}st('exported '+d.n+' object(s) -> '+d.dir)});}
</script></body></html>"""


class _VizEngine(GeoLangSplatEngine):
    """Engine + a live fvdb.viz scene that recolours / cuts out on each request."""

    def attach_scene(self, scene):
        self.scene = scene
        self._catalog = None
        self._cat_word = ""
        self._show(torch.zeros(self.N, dtype=torch.bool, device=self.device))

    def _show(self, sel: torch.Tensor) -> None:
        """Show the full scene with ``sel`` Gaussians tinted (everything stays)."""
        from fvdb import GaussianSplat3d

        cfg = self.cfg
        sh0 = self.model.sh0.detach().clone()
        shN = self.model.shN.detach().clone()
        if int(sel.sum()) > 0:
            a = float(cfg.blend)
            cur = 0.5 + SH_C0 * sh0[sel]
            tgt = torch.tensor(cfg.highlight_color, device=self.device).view(1, 1, 3)
            sh0[sel] = ((1 - a) * cur + a * tgt - 0.5) / SH_C0
            shN[sel] = shN[sel] * (1 - a)
        disp = GaussianSplat3d.from_tensors(
            means=self.means,
            quats=self.model.quats.detach(),
            log_scales=self.model.log_scales.detach(),
            logit_opacities=self.model.logit_opacities.detach(),
            sh0=sh0,
            shN=shN,
        )
        # Re-add under the same name to swap in place (no scene.reset(), which would
        # snap the camera back to center).
        self.scene.add_gaussian_splat_3d("scene", disp)

    def _show_labels(self, labels: torch.Tensor) -> None:
        """Recolour the full scene by object id -- each instance gets its own colour."""
        from fvdb import GaussianSplat3d

        cfg = self.cfg
        sh0 = self.model.sh0.detach().clone()
        shN = self.model.shN.detach().clone()
        labels = labels.to(self.device)
        ids = [int(x) for x in torch.unique(labels).tolist() if int(x) >= 0]
        a = max(float(cfg.blend), 0.85)
        for color, oid in zip(_object_palette(len(ids)), ids):
            m = labels == oid
            tgt = torch.tensor(color, device=self.device, dtype=sh0.dtype).view(1, 1, 3)
            cur = 0.5 + SH_C0 * sh0[m]
            sh0[m] = ((1 - a) * cur + a * tgt - 0.5) / SH_C0
            shN[m] = shN[m] * (1 - a)
        disp = GaussianSplat3d.from_tensors(
            means=self.means,
            quats=self.model.quats.detach(),
            log_scales=self.model.log_scales.detach(),
            logit_opacities=self.model.logit_opacities.detach(),
            sh0=sh0,
            shN=shN,
        )
        self.scene.add_gaussian_splat_3d("scene", disp)

    def _show_cutout(self, mask: torch.Tensor) -> None:
        """Swap the scene to *only* the masked Gaussians (a clean object cutout).

        Keeps full spherical harmonics (no quality loss) and reframes the camera
        tight on the object so it's easy to orbit/zoom -- otherwise the camera stays
        parked at the whole-scene distance and the cutout looks tiny and far.
        """
        from fvdb import GaussianSplat3d

        m = mask.to(self.device)
        if int(m.sum()) == 0:
            return
        disp = GaussianSplat3d.from_tensors(
            means=self.means[m],
            quats=self.model.quats.detach()[m],
            log_scales=self.model.log_scales.detach()[m],
            logit_opacities=self.model.logit_opacities.detach()[m],
            sh0=self.model.sh0.detach()[m],
            shN=self.model.shN.detach()[m],
        )
        self.scene.add_gaussian_splat_3d("scene", disp)
        self._frame_on(self.means[m])

    def _frame_on(self, pts: torch.Tensor) -> None:
        """Point the viewer camera at ``pts`` from close range (tight bbox framing)."""
        import numpy as np

        from .. import autoview
        from ..cameras import orbit_cameras

        try:
            up = autoview.resolve_up(getattr(self.cfg, "up", "auto"), self.means)
            m = pts.detach().float().cpu().numpy()
            lo, hi = m.min(axis=0), m.max(axis=0)
            center = (lo + hi) / 2
            radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-3)
            self.scene.camera_up_direction = up
            self.scene.camera_near = max(radius * 0.002, 1e-4)
            self.scene.camera_far = radius * 100
            c2w = orbit_cameras(center, 25, radius * 2.0, num_azimuth=1, azimuth_offset_deg=35, up=up)[0][1]
            self.scene.set_camera_lookat(c2w[:3, 3], center, up)
        except Exception as e:  # pragma: no cover - viewer runtime only
            print(f"[viewer] could not frame object: {e}", flush=True)

    # -- catalog flow -------------------------------------------------------

    def viz_catalog(self, word, *, select=None, gran=None, min_size=None) -> dict:
        """Catalog one class and light up every instance in its own colour."""
        w = str(word).strip().lower()
        if not w:
            raise ValueError("type one class word")
        cat = self.catalog(
            [w],
            select=select,
            link_frac=float(gran) if gran else None,
            min_size=int(min_size) if min_size else None,
        )
        self._catalog = cat
        self._cat_word = w
        self._show_labels(cat.labels)
        rows = [{"id": o.id, "n": o.n_gaussians} for o in cat.objects]
        return {"word": w, "n": len(cat), "rows": rows}

    def viz_cutout(self, obj_id: int) -> dict:
        """Swap the viewer to a cutout of just object ``obj_id``."""
        if self._catalog is None:
            raise RuntimeError("search a class first")
        sel = self._catalog.mask(obj_id)
        self._show_cutout(sel)
        return {"id": int(obj_id), "n": int(sel.sum())}

    def viz_back(self) -> dict:
        """Return from a cutout to the all-instances highlighted view (wide frame)."""
        if self._catalog is None:
            raise RuntimeError("nothing to go back to")
        self._show_labels(self._catalog.labels)
        _frame_viewer(self.scene, self, self.cfg)
        return {"ok": True}

    def viz_export(self, out_dir: str) -> dict:
        """Write the current catalog (per-object .ply + catalog.csv) to ``out_dir``."""
        if self._catalog is None:
            raise RuntimeError("search a class first")
        path = self._catalog.export_all(out_dir)
        return {"dir": str(path), "n": len(self._catalog)}


def _make_handler(engine: _VizEngine):
    cfg = engine.cfg

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="text/html"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _try(self, fn):
            try:
                self._json(200, fn())
            except Exception as e:  # pragma: no cover - viewer runtime only
                self._json(500, {"error": str(e)})

        def do_GET(self):
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            if parsed.path == "/catalog":
                word = qs.get("q", [""])[0]
                select = float(qs.get("select", [cfg.select])[0])
                gran = qs.get("gran", [None])[0]
                min_size = qs.get("min", [None])[0]
                self._try(lambda: engine.viz_catalog(word, select=select, gran=gran, min_size=min_size))
                return
            if parsed.path == "/cutout":
                self._try(lambda: engine.viz_cutout(int(qs.get("id", ["0"])[0])))
                return
            if parsed.path == "/back":
                self._try(engine.viz_back)
                return
            if parsed.path == "/export":
                out_dir = qs.get("dir", ["catalog"])[0]
                self._try(lambda: engine.viz_export(out_dir))
                return
            html = (
                PAGE.replace("__VIEWPORT__", str(cfg.viewer_port))
                .replace("__CHIPS__", json.dumps(list(_CHIPS)))
                .replace("__SELECT__", str(cfg.select))
            )
            self._send(200, html.encode())

    return H


def run_viewer(model_path: str, config: GeoLangSplatConfig | None = None, recipe: str | None = None) -> None:
    """Launch the fvdb viewer + the web catalog UI (blocks)."""
    import fvdb.viz as viz

    cfg = config or GeoLangSplatConfig()
    if recipe:
        apply_recipe(cfg, recipe)
    viz.init(ip_address="0.0.0.0", port=cfg.viewer_port, vk_device_id=cfg.vk_device_id)
    engine = _VizEngine(model_path, cfg)
    scene = viz.get_scene("GeoLangSplat")
    engine.attach_scene(scene)
    _frame_viewer(scene, engine, cfg)
    httpd = HTTPServer(("0.0.0.0", cfg.web_port), _make_handler(engine))
    print(f"[CATALOG UI] http://<host>:{cfg.web_port}   [viewer] http://<host>:{cfg.viewer_port}", flush=True)
    httpd.serve_forever()
