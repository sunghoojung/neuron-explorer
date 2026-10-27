import concurrent.futures as cf
import os
import re
import sys
import urllib.parse
import urllib.request

BASE = "https://already-clean.goodfire.com/"
OUT = sys.argv[1]
TEXT_EXT = ('.html','.css','.js','.mjs','.json','.md','.svg','.txt')
ref_re = re.compile(r'''(?:src|href|data-src|poster|srcset)\s*=\s*["']([^"']+)["']|url\(\s*["']?([^"')]+)["']?\s*\)|["'`]((?:\.{0,2}/)?(?:assets|data|microscope|img|images|media)/[^"'`\s<>]+)["'`]|["'`]([\w./-]+\.(?:js|mjs|css|json|png|jpe?g|webp|gif|svg|mp4|webm|md|bin|npy|txt))["'`]''')
seen, fails = set(), []
def norm(ref, parent):
    ref = ref.strip().split('#')[0]
    if not ref or ref.startswith(('data:','mailto:','javascript:','blob:')) or '${' in ref or '{' in ref: return None
    u = urllib.parse.urljoin(parent, ref)
    p = urllib.parse.urlparse(u)
    if p.netloc != urllib.parse.urlparse(BASE).netloc: return None
    return urllib.parse.urlunparse((p.scheme,p.netloc,p.path,'','',''))
def local(u):
    path = urllib.parse.urlparse(u).path.lstrip('/') or 'index.html'
    if path.endswith('/'): path += 'index.html'
    return os.path.join(OUT, urllib.parse.unquote(path))
def fetch(u):
    lp = local(u)
    try:
        if not os.path.exists(lp):
            req = urllib.request.Request(u, headers={'User-Agent':'Mozilla/5.0'})
            data = urllib.request.urlopen(req, timeout=60).read()
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            with open(lp, 'wb') as handle:
                handle.write(data)
        if lp.endswith(TEXT_EXT) or lp.endswith('index.html'):
            with open(lp, encoding='utf-8', errors='ignore') as handle:
                txt = handle.read()
            refs = set()
            for m in ref_re.finditer(txt):
                for g in m.groups():
                    if not g: continue
                    for part in g.split(','):  # srcset
                        for par in ((u, BASE) if lp.endswith(('.js','.mjs','.json')) else (u,)):
                            r = norm(part.strip().split(' ')[0], par)
                            if r: refs.add(r)
            return u, refs, None
        return u, set(), None
    except Exception as e:  # noqa: BLE001 - record each fetch failure and continue the crawl
        return u, set(), str(e)
frontier = {BASE}
with cf.ThreadPoolExecutor(24) as ex:
    while frontier:
        seen |= frontier
        nxt = set()
        for u, refs, err in ex.map(fetch, frontier):
            if err: fails.append((u, err))
            nxt |= refs - seen
        print(f"fetched {len(frontier)}, total {len(seen)}, next {len(nxt)}, fails {len(fails)}", flush=True)
        frontier = nxt
with open(os.path.join(OUT, '_fails.txt'), 'w') as handle:
    handle.write('\n'.join(f"{u}\t{e}" for u,e in fails))
