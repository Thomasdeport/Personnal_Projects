"""Publish this code folder as a PRIVATE Kaggle Dataset (new version on each run).

    python scripts/kaggle_upload.py                      # handle = <your username>/cfm-signature-lab
    python scripts/kaggle_upload.py --notes "ablation venues"
    python scripts/kaggle_upload.py --dry-run            # list what would be uploaded, upload nothing

Credentials (once, never committed): Kaggle → Settings → API → "Create New Token", then either
    ~/.kaggle/kaggle.json (chmod 600)   or   export KAGGLE_USERNAME=... KAGGLE_KEY=...
kagglehub creates new datasets as private. The version note records the git commit and dirty state.
"""
import argparse
import fnmatch
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'cfm-signature-lab'
IGNORE = ['.git/', '__pycache__/', '.ipynb_checkpoints/', 'demo_lab/', 'demo_data/', 'data/', 'lab/',
          '*.npy', '*.npz', '*.pt', '*.csv', '*.zip', '.DS_Store']


def ignored(rel):
    parts = rel.split('/')
    for pat in IGNORE:
        if pat.endswith('/'):
            if pat[:-1] in parts[:-1]:
                return True
        elif fnmatch.fnmatch(parts[-1], pat):
            return True
    return False


def git_note():
    try:
        sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return f'git {sha}' + (' + local changes' if dirty else '')
    except OSError:
        return 'no git'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--handle', help=f'owner/slug (default: <kaggle username>/{SLUG})')
    p.add_argument('--notes', default='')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    files = sorted(str(f.relative_to(ROOT)) for f in ROOT.rglob('*') if f.is_file())
    keep = [f for f in files if not ignored(f)]
    print(f'{len(keep)} files to upload ({len(files) - len(keep)} ignored)')
    if not (ROOT / 'cfm' / '__init__.py').exists() or not (ROOT / 'CFM_Signature_Lab.ipynb').exists():
        raise SystemExit('Run from the CFM repo: cfm/ or CFM_Signature_Lab.ipynb is missing')
    if a.dry_run:
        print('\n'.join(keep)); return
    import kagglehub
    handle = a.handle or f'{kagglehub.whoami()["username"]}/{SLUG}'
    notes = ' | '.join(x for x in [git_note(), a.notes] if x)
    kagglehub.dataset_upload(handle, str(ROOT), version_notes=notes, ignore_patterns=IGNORE)
    print(f'Uploaded → https://www.kaggle.com/datasets/{handle}  ({notes})')
    print(f"In the Kaggle notebook: CODE_DATASET = '{handle}'")


if __name__ == '__main__':
    main()
