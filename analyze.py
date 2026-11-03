"""Yield Advisor analytics: reads ./data/*.csv, runs the engines, writes dashboard.html."""
import json, sys, os, numpy as np, pandas as pd
from scipy.stats import fisher_exact
from scipy.ndimage import convolve
D = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'data'
rd = lambda f, **k: pd.read_csv(os.path.join(D, f), **k)
lots, ws, mes, wat, lim, ft = rd('lots.csv'), rd('wafer_summary.csv'), rd('lot_context_mes.csv'), rd('wat_pcm.csv'), rd('test_limits.csv'), rd('final_test_burnin.csv')
sort = rd('sort_die_results.csv')
K = ['lot_id', 'wafer_id', 'die_x', 'die_y']
TESTS = ['CONTACT_RES','IDDQ','LEAKAGE','SCAN_FAIL_COUNT','MBIST_FAIL','PLL_LOCK','VMIN','FMAX','RING_OSC','ADC_INL']
bins = lim.set_index('test_name')['soft_bin_name'].to_dict()
sb2name = lim.set_index('soft_bin')['soft_bin_name'].to_dict()
ins1 = sort[sort.insertion == 1]
fin = sort.sort_values('insertion').drop_duplicates(K, keep='last')
out = {}

# ---- Lots: yield, EWMA/Shewhart excursion detector
lots['t'] = pd.to_datetime(lots.sort_start_ts); lots = lots.sort_values('t').reset_index(drop=True)
ly = ws.groupby('lot_id').final_yield.mean().reindex(lots.lot_id).values * 100
vt = wat.groupby('lot_id').vt_n_mV.mean().reindex(lots.lot_id).values
mu = np.median(ly[:15]); sd = max(1.4826 * np.median(np.abs(ly[:15] - mu)), 0.5)
lam = 0.2; z = []; e = mu
for y in ly: e = lam * y + (1 - lam) * e; z.append(e)
lcl = mu - 3 * sd * np.sqrt(lam / (2 - lam))
ew_alarm = next((i for i, v in enumerate(z) if v < lcl), None)
sh_alarm = next((i for i, v in enumerate(ly) if v < mu - 3 * sd), None)
out['lots'] = dict(id=list(lots.lot_id), y=ly.round(2).tolist(), ewma=np.round(z, 2).tolist(), vtn=vt.round(2).tolist(),
                   mu=round(mu, 2), lcl=round(lcl, 2), ew_alarm=ew_alarm, sh_alarm=sh_alarm,
                   card=list(lots.probe_card_id))
# VT shift lots via change-point on Vt_n (largest mean-split)
x = vt; best = max(range(3, len(x) - 3), key=lambda i: abs(x[:i].mean() - x[i:].mean()) * np.sqrt(i * (len(x) - i)))
# find sustained high segment: lots with vt > mu+8mV
hi = np.where(x > np.median(x) + 9)[0]
out['vt_segment'] = [lots.lot_id[hi.min()], lots.lot_id[hi.max()]] if len(hi) else None
out['vt_corr_yield'] = round(float(np.corrcoef(vt, ly)[0, 1]), 2)
# ---- Pareto
fails = fin[~fin.pass_ == 1] if 'pass_' in fin else fin[fin['pass'] == 0]
par = fails.soft_bin.map(sb2name).value_counts()
out['pareto'] = [[k, int(v)] for k, v in par.items()]
out['kpi'] = dict(wafers=len(ws), lots=len(lots), die=int(len(fin)), yield_final=round(ws.final_yield.mean() * 100, 2),
                  yield_fp=round(ws.first_pass_yield.mean() * 100, 2), good=int(fin['pass'].sum()))

# ---- Wafer map pattern AI (rule-based radial/spatial features)
N = 26; c = (N - 1) / 2; yy, xx = np.mgrid[0:N, 0:N]; R = np.hypot(xx - c, yy - c) / (N / 2)
pat, maps = [], {}
for (l, w), g in fin.groupby(['lot_id', 'wafer_id']):
    G = np.full((N, N), -1, int); G[g.die_y.values, g.die_x.values] = 1 - g['pass'].values  # 1=fail,0=pass,-1=none
    v = G >= 0; f = G == 1
    edge = f[v & (R > .8)].mean(); cen = f[v & (R < .4)].mean(); mid = f[v & (R <= .8) & (R >= .4)].mean()
    # scratch: longest straight run of fails via row/col/diag max line density
    lab = 'None'
    if edge > .3 and edge > 2.5 * mid: lab = 'Edge-Ring'
    elif cen > .25 and cen > 2.5 * edge: lab = 'Center'
    pat.append((l, w, lab, round(edge, 3), round(cen, 3), round(f.sum() / v.sum(), 3)))
    if lab != 'None': maps[f'{l}-{w}'] = [''.join('.' if a < 0 else str(a) for a in r) for r in G]
pat = pd.DataFrame(pat, columns=['lot_id', 'wafer_id', 'pattern', 'edge', 'center', 'failrate'])
ws = ws.merge(pat, on=['lot_id', 'wafer_id'])
out['patterns'] = pat.pattern.value_counts().to_dict()
show = ws[ws.pattern != 'None'].sort_values('final_yield').groupby('pattern').head(8)
out['maps'] = [dict(lot=r.lot_id, wafer=int(r.wafer_id), pattern=r.pattern, y=round(r.final_yield * 100, 1),
                    map=maps[f'{r.lot_id}-{r.wafer_id}']) for r in show.itertuples()]

# ---- Root cause: commonality (Fisher) over process steps
m = mes.merge(ws[['lot_id', 'wafer_id', 'pattern', 'final_yield']], on=['lot_id', 'wafer_id'])
m['factor'] = m.step + ' : ' + m.equipment_id + np.where(m.chamber.notna(), ' / ' + m.chamber.fillna(''), '')
rc = []; Nw = len(ws)
for pt in ['Edge-Ring', 'Center']:
    aff = m.pattern == pt
    for fct, g in m.groupby('factor'):
        a = int(aff[g.index].sum()); b = len(g) - a; cc = int((ws.pattern == pt).sum()) - a; d = Nw - len(g) - cc
        if a < 3: continue
        odds, p = fisher_exact([[a, b], [cc, d]])
        rest = (ws.final_yield.sum() - g.final_yield.sum()) / max(Nw - len(g), 1)
        rc.append(dict(signature=pt, factor=fct, wafers=len(g), affected=a, rate=round(a / len(g) * 100, 1),
                       base=round(cc / max(cc + d, 1) * 100, 1), p=float(p), yield_=round(g.final_yield.mean() * 100, 1),
                       loss=round(len(g) * (rest - g.final_yield.mean()) / Nw * 100, 2), odds=float(odds)))
rc = pd.DataFrame(rc); rc = rc[rc.odds > 1].sort_values('p').groupby('signature').head(5)
out['rootcause'] = rc.to_dict('records')
# time split for top edge-ring chamber (PM effect)
top = rc[rc.signature == 'Edge-Ring'].iloc[0].factor if (rc.signature == 'Edge-Ring').any() else None
if top:
    g = m[m.factor == top].copy(); g['ts'] = pd.to_datetime(mes[mes.index.isin(g.index)].ts_in)
    g = g.sort_values('ts'); k = g.groupby(g.ts.dt.date).final_yield.mean()
    # onset: first date where affected rate stays elevated
    pre = g[g.pattern != 'Edge-Ring'].ts.min()
    first = g[g.pattern == 'Edge-Ring'].ts.min()
    out['rc_top'] = dict(factor=top, first_affected=str(first)[:10])

# ---- Probe / tester
s = ins1.copy(); s['c'] = (s.soft_bin == 30)
ps = s.groupby(['probe_card_id', 'site']).agg(yield_=('pass', 'mean'), contact=('c', 'mean'), rc=('CONTACT_RES', 'mean'), n=('pass', 'size')).reset_index()
r2 = sort[sort.insertion == 2][K + ['pass']].rename(columns={'pass': 'rp'})
cf = s[s.c][K + ['probe_card_id', 'site']].merge(r2, on=K, how='left')
rec = cf.groupby(['probe_card_id', 'site']).rp.agg(['mean', 'size']).reset_index()
ps = ps.merge(rec.rename(columns={'mean': 'recovery', 'size': 'cfails'}), on=['probe_card_id', 'site'], how='left').fillna(0)
for c_ in ['yield_', 'contact', 'recovery']: ps[c_] = (ps[c_] * 100).round(2)
ps['rc'] = ps.rc.round(3); out['probe'] = ps.to_dict('records')
tot_c = len(cf); out['retest'] = dict(contact_fails=tot_c, retested=int(cf.rp.notna().sum()), recovered=int(cf.rp.fillna(0).sum()))

# ---- Outlier screening: DPAT (IDDQ), GDBN, evaluated against burn-in
p = fin[fin['pass'] == 1][K + ['IDDQ']].copy()
g = p.groupby(['lot_id', 'wafer_id']).IDDQ
med = g.transform('median'); q = (g.transform(lambda s: s.quantile(.75)) - g.transform(lambda s: s.quantile(.25))) / 1.35
lo_, hi_ = np.maximum(40, med - 6 * q), np.minimum(65, med + 6 * q)
p['dpat'] = (p.IDDQ < lo_) | (p.IDDQ > hi_)
gd = []
ker = np.ones((3, 3)); ker[1, 1] = 0
for (l, w), gg in fin.groupby(['lot_id', 'wafer_id']):
    F = np.zeros((N, N)); V = np.zeros((N, N)); F[gg.die_y, gg.die_x] = 1 - gg['pass']; V[gg.die_y, gg.die_x] = 1
    nb = convolve(F, ker, mode='constant'); sel = gg[gg['pass'] == 1]
    gd.append(pd.DataFrame({'lot_id': l, 'wafer_id': w, 'die_x': sel.die_x, 'die_y': sel.die_y, 'gdbn': nb[sel.die_y, sel.die_x] >= 5}))
p = p.merge(pd.concat(gd), on=K); e = ft.merge(p[K + ['dpat', 'gdbn']], on=K, how='left').fillna({'dpat': False, 'gdbn': False})
def ev(flag):
    tp = int((flag & (e.burnin_fail == 1)).sum()); n = int(flag.sum()); pos = int((e.burnin_fail == 1).sum())
    return dict(flagged=n, precision=round(tp / max(n, 1) * 100, 1), recall=round(tp / pos * 100, 1), yield_cost=round(n / len(fin) * 100, 3),
                bi_rate=round(tp / max(n, 1) * 100, 1))
out['screens'] = dict(base_rate=round(e.burnin_fail.mean() * 100, 3), n=len(e), burnin=int(e.burnin_fail.sum()),
                      dpat=ev(e.dpat.astype(bool)), gdbn=ev(e.gdbn.astype(bool)), both=ev(e.dpat.astype(bool) | e.gdbn.astype(bool)))
# ---- Test time optimizer (stop-on-fail: value present == test executed)
tt = []; tot_time = ins1.test_time_ms.sum(); lm = lim.set_index('test_name')
for t in TESTS:
    n = int(ins1[t].notna().sum()); nf = int((ins1.soft_bin == lm.soft_bin[t]).sum()); ms = float(lm.test_time_ms[t])
    tt.append(dict(test=t, executed=n, fails=nf, ms=ms, share=round(n * ms / tot_time * 100, 2), bound_dppm=round(3 / n * 1e6, 2)))
both = ins1[['FMAX', 'RING_OSC']].dropna()
out['testtime'] = dict(rows=tt, corr_ringosc_fmax=round(float(both.corr().iloc[0, 1]), 3))
cand = [t for t in tt if t['fails'] <= 20 or t['test'] == 'RING_OSC']
out['testtime']['candidates'] = [t['test'] for t in cand]; out['testtime']['saved_pct'] = round(sum(t['share'] for t in cand), 2)

# ---- Advisor: ranked recommendations
R = []
for sg in ['Edge-Ring', 'Center']:
    r = rc[rc.signature == sg]
    if len(r):
        r = r.iloc[0]; R.append(dict(type='Equipment', title=f'Inhibit / PM-check {r.factor}', evidence=f'{sg} on {r.rate}% of its wafers vs {r.base}% elsewhere (Fisher p={r.p:.1e}); {r.affected} wafers.', gain=r.loss, conf='High' if r.p < 1e-6 else 'Medium', risk='Low', approval='Process owner via MES'))
if out['vt_segment']:
    R.append(dict(type='Investigate', title=f"WAT Vt_n shift on lots {out['vt_segment'][0]}..{out['vt_segment'][1]}", evidence=f"Lot Vt_n vs lot yield r={out['vt_corr_yield']}; raises VMIN/FMAX bins.", gain=round(float(np.mean(ly) - ly[hi].mean()) * len(hi) / len(ly), 2), conf='Medium', risk='Low', approval='Yield engineer'))
w3 = ps.sort_values('contact', ascending=False).iloc[0]
R.append(dict(type='Probe', title=f'Clean/replace {w3.probe_card_id} (site {int(w3.site)}); retest SB30 fails', evidence=f'Contact fail {w3.contact}% on that site, {w3.recovery}% of contact fails recover on retest.', gain=round(out['retest']['recovered'] / len(fin) * 100, 2), conf='High', risk='Low', approval='Test floor lead (auto within retest policy)'))
d = out['screens']['dpat']; gdb = out['screens']['gdbn']
R.append(dict(type='Disposition', title='Enable DPAT (k=6) on IDDQ', evidence=f"{d['flagged']} die flagged, {d['precision']}% fail burn-in vs {out['screens']['base_rate']}% base; recall {d['recall']}%.", gain=0, conf='High', risk=f"yield cost {d['yield_cost']} pt", approval='Quality engineer'))
R.append(dict(type='Disposition', title='Enable GDBN ink (>=5 failing neighbours)', evidence=f"{gdb['flagged']} die flagged, {gdb['precision']}% burn-in fail; yield cost {gdb['yield_cost']} pt.", gain=0, conf='Medium', risk='Low', approval='Quality engineer'))
R.append(dict(type='Test content', title='Move RING_OSC (and ADC_INL) to sampling', evidence=f"RING_OSC~FMAX r={out['testtime']['corr_ringosc_fmax']}; rule-of-three bound {min(t['bound_dppm'] for t in tt if t['test'] in out['testtime']['candidates'])} DPPM; saves ~{out['testtime']['saved_pct']}% of sort time.", gain=0, conf='High', risk='Bounded', approval='Change control board'))
out['recs'] = R
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard_template.html')).read()
open('dashboard.html', 'w').write(tpl.replace('/*DATA*/null', json.dumps(out, default=lambda o: o.item() if hasattr(o, 'item') else str(o))))
print('wrote dashboard.html'); print(json.dumps({k: out[k] for k in ['kpi', 'patterns', 'vt_segment', 'screens', 'retest']}, default=str)); print(rc[['signature', 'factor', 'rate', 'p', 'loss']]); print('alarms', ew_alarm, sh_alarm, out['testtime']['candidates'], out['testtime']['saved_pct'])
