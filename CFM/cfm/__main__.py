"""Command line. Every command takes --config and repeated --set key.path=value overrides.

python -m cfm prepare   --config configs/base.json
python -m cfm blocks
python -m cfm screen    univariate|adversarial|forward|ablate|greedy --config ...
python -m cfm tabular   --blocks base_relative cat_freq ticks --model xgb [--refit]
python -m cfm neural    --config configs/signature.json --name sig_s0 [--refit epochs|steps]
python -m cfm blend     --runs sig_s0 tab_xgb_x [--optimise] [--submit --source test_refit] [--balance]
python -m cfm audit     --runs ... --confirm
python -m cfm synthetic --out data/synth
"""
import argparse
import json
from pathlib import Path
from . import config as C


def prepare(cfg, files=None):
    from . import io, registry, split
    from . import features  # noqa: F401
    files = files or io.find_files(cfg['data_root'], **{k: v for k, v in cfg['files'].items() if v})
    meta = io.build(files, cfg['lab_dir'], cfg['tick'])
    print('tick report:', json.dumps(meta['tick_report']), flush=True)
    if not meta['tick_report'].get('grid_ok', True):
        print('WARNING: prices are not on the estimated tick grid → tick-based blocks are approximations.')
    for name in registry.BLOCKS:
        for s in ['train', 'test']:
            registry.compute(name, cfg['lab_dir'], s, cfg['lot'])
    return split.make(cfg['lab_dir'], cfg)


def main(argv=None):
    p = argparse.ArgumentParser(prog='cfm')
    p.add_argument('command')
    p.add_argument('stage', nargs='?')
    p.add_argument('--config', default='configs/base.json')
    p.add_argument('--set', action='append', default=[])
    p.add_argument('--blocks', nargs='*')
    p.add_argument('--model', default='xgb')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--name')
    p.add_argument('--refit', nargs='?', const='epochs')
    p.add_argument('--runs', nargs='*')
    p.add_argument('--optimise', action='store_true')
    p.add_argument('--submit', action='store_true')
    p.add_argument('--source', default='test_refit')
    p.add_argument('--balance', action='store_true')
    p.add_argument('--confirm', action='store_true')
    p.add_argument('--out', default='data/synth')
    p.add_argument('--note', default='')
    a = p.parse_args(argv)
    if a.command == 'synthetic':
        from .synthetic import create
        print(create(a.out)); return
    cfg = C.load(a.config, a.set)
    lab = cfg['lab_dir']
    if a.command == 'prepare':
        prepare(cfg)
    elif a.command == 'blocks':
        from .registry import catalogue
        print(catalogue().to_string(index=False))
    elif a.command == 'screen':
        from . import screen, plots
        s = cfg['screen']
        blocks = a.blocks or s['candidates']
        if a.stage == 'univariate':
            df = screen.univariate(lab, cfg, list(s['base']) + blocks, base=s['base']); plots.screen_map(df, lab)
            print(df.head(30).to_string(index=False))
        elif a.stage == 'adversarial':
            plots.adversarial_bars(screen.adversarial(lab, cfg, list(s['base']) + blocks), lab)
        elif a.stage == 'forward':
            plots.forward_bars(screen.forward(lab, cfg, s['base'], blocks, s['model'], None, s['seeds'], s['fit_frac']), lab)
        elif a.stage == 'ablate':
            screen.ablate(lab, cfg, list(s['base']) + blocks, s['model'], None, s['seeds'], s['fit_frac'])
        elif a.stage == 'greedy':
            print(screen.greedy(lab, cfg, s['base'], blocks, s['model'], None, s['seeds'][0], s['fit_frac'])[1])
        else:
            raise SystemExit('screen stage: univariate | adversarial | forward | ablate | greedy')
    elif a.command == 'tabular':
        from .tabular import run
        run(lab, cfg, a.blocks, a.model, seed=a.seed, name=a.name, refit=bool(a.refit), note=a.note)
    elif a.command == 'neural':
        from .neural.train import fit, refit
        name = a.name or Path(a.config).stem
        fit(lab, cfg, name, note=a.note)
        if a.refit:
            refit(lab, cfg, name, a.refit)
    elif a.command == 'blend':
        from . import blend, plots
        table, w, agree, oracle = blend.report(lab, a.runs, a.optimise)
        print(table.round(4).to_string(index=False)); print('weights', w.round(3), '| oracle', round(oracle, 4))
        plots.agreement(agree, lab)
        if a.submit:
            print(blend.submission(lab, a.runs, w, a.source, a.balance))
    elif a.command == 'audit':
        from .audit import open_audit
        print(json.dumps(open_audit(lab, a.runs, confirm=a.confirm), indent=2))
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
