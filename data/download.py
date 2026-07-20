"""Fetch DIV2K (train/val), Kodak, and McMaster into ./datasets/.

Usage:
    python -m data.download                     # everything
    python -m data.download --datasets kodak mcm
    python -m data.download --datasets div2k_val kodak --root ./datasets

If a source is unavailable, the expected folder layout is printed so files
can be dropped in manually.
"""
import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

LAYOUT = """
Expected folder layout under --root (default ./datasets):

  datasets/
    DIV2K_train_HR/   0001.png ... 0800.png     (DIV2K train, 800 images)
    DIV2K_valid_HR/   0801.png ... 0900.png     (DIV2K val, 100 images)
    Kodak/            kodim01.png ... kodim24.png
    McM/              1.tif ... 18.tif  (or .png; 18 images, 500x500)

Sources if you need to fetch manually:
  DIV2K:  https://data.vision.ee.ethz.ch/cvl/DIV2K/
  Kodak:  https://r0k.us/graphics/kodak/
  McM:    https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm
"""

DIV2K = {
    # HF mirror first (often much faster than the ETH origin), then origin.
    'div2k_train': ('DIV2K_train_HR',
                    ['https://huggingface.co/datasets/yangtao9009/DIV2K/resolve/main/DIV2K_train.zip',
                     'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip']),
    'div2k_val': ('DIV2K_valid_HR',
                  ['https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip']),
}

KODAK_URL = 'https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png'

# Official McMaster/IMAX source (Zhang et al.); the zip is password-protected
# with the password published on the dataset page itself.
MCM_URL = 'https://www4.comp.polyu.edu.hk/~cslzhang/DATA/McM.zip'
MCM_PWD = b'McM_CDM'


_last_pct = [-1.0]


def _progress(block_num, block_size, total_size):
    if total_size > 0:
        done = min(block_num * block_size, total_size)
        pct = 100.0 * done / total_size
        if pct - _last_pct[0] >= 1.0 or done >= total_size:
            _last_pct[0] = pct
            sys.stdout.write(f'\r  {done / 1e6:8.1f}/{total_size / 1e6:.1f} MB ({pct:5.1f}%)')
            sys.stdout.flush()


def fetch(url, dst):
    print(f'  downloading {url}')
    tmp = dst + '.part'
    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    print()
    os.replace(tmp, dst)


def fetch_zip(urls, out_dir, member_dir=None):
    """Try urls in order; unzip into out_dir (flattening member_dir if given)."""
    os.makedirs(out_dir, exist_ok=True)
    zpath = out_dir.rstrip('/\\') + '.zip'
    last_err = None
    for url in urls:
        try:
            if not os.path.exists(zpath):
                fetch(url, zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(os.path.dirname(out_dir) or '.')
            # If the zip created a differently-named folder, move contents.
            if n_images(out_dir) == 0:
                with zipfile.ZipFile(zpath) as z:
                    tops = {m.split('/')[0] for m in z.namelist() if '/' in m}
                root = os.path.dirname(out_dir) or '.'
                for top in tops:
                    cand = os.path.join(root, top)
                    if n_images(cand) > 0 and \
                            os.path.abspath(cand) != os.path.abspath(out_dir):
                        for f in os.listdir(cand):
                            shutil.move(os.path.join(cand, f), out_dir)
                        os.rmdir(cand)
            os.remove(zpath)
            return True
        except Exception as e:  # noqa: BLE001 - report and try next mirror
            last_err = e
            print(f'  FAILED: {e}')
            if os.path.exists(zpath):
                os.remove(zpath)
    print(f'  could not fetch from any source (last error: {last_err})')
    return False


def n_images(d):
    if not os.path.isdir(d):
        return 0
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    return sum(1 for f in os.listdir(d) if os.path.splitext(f)[1].lower() in exts)


def get_div2k(root, which):
    sub, urls = DIV2K[which]
    out = os.path.join(root, sub)
    if n_images(out) > 0:
        print(f'[{which}] already present ({n_images(out)} images) - skipping')
        return True
    print(f'[{which}] -> {out}')
    ok = fetch_zip(urls, out, member_dir=os.path.join(root, sub))
    print(f'[{which}] {n_images(out)} images')
    return ok


def get_kodak(root):
    out = os.path.join(root, 'Kodak')
    if n_images(out) >= 24:
        print(f'[kodak] already present (24 images) - skipping')
        return True
    os.makedirs(out, exist_ok=True)
    print(f'[kodak] -> {out}')
    ok = True
    for i in range(1, 25):
        dst = os.path.join(out, f'kodim{i:02d}.png')
        if os.path.exists(dst):
            continue
        try:
            fetch(KODAK_URL.format(i), dst)
        except Exception as e:  # noqa: BLE001
            print(f'  FAILED kodim{i:02d}: {e}')
            ok = False
    print(f'[kodak] {n_images(out)} images')
    return ok


def get_mcm(root):
    out = os.path.join(root, 'McM')
    if n_images(out) >= 18:
        print(f'[mcm] already present ({n_images(out)} images) - skipping')
        return True
    os.makedirs(out, exist_ok=True)
    print(f'[mcm] -> {out}')
    zpath = os.path.join(root, 'McM.zip')
    try:
        if not os.path.exists(zpath):
            fetch(MCM_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                name = os.path.basename(info.filename)
                if os.path.splitext(name)[1].lower() not in \
                        {'.png', '.tif', '.tiff', '.bmp'}:
                    continue
                with z.open(info, pwd=MCM_PWD) as src, \
                        open(os.path.join(out, name), 'wb') as dst:
                    shutil.copyfileobj(src, dst)
        os.remove(zpath)
    except Exception as e:  # noqa: BLE001
        print(f'  FAILED: {e}')
    print(f'[mcm] {n_images(out)} images')
    return n_images(out) >= 18


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='./datasets')
    p.add_argument('--datasets', nargs='+',
                   default=['div2k_train', 'div2k_val', 'kodak', 'mcm'],
                   choices=['div2k_train', 'div2k_val', 'kodak', 'mcm'])
    args = p.parse_args()
    os.makedirs(args.root, exist_ok=True)

    all_ok = True
    for d in args.datasets:
        if d in DIV2K:
            all_ok &= get_div2k(args.root, d)
        elif d == 'kodak':
            all_ok &= get_kodak(args.root)
        elif d == 'mcm':
            all_ok &= get_mcm(args.root)
    if not all_ok:
        print(LAYOUT)
        sys.exit(1)
    print('done.')


if __name__ == '__main__':
    main()
