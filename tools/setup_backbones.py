"""Fetch pinned official code/model assets; never overwrite an existing weight."""
import argparse,shutil,sys,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from zip2cad.release_io import read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('component',choices=['zipmap','sonata','mobilesam']);p.add_argument('--weights',action='store_true');a=p.parse_args()
    item=read(ROOT/'models/manifest.json')[a.component];target=ROOT/item['repository_directory']
    if target.exists():print('Source already present:',target)
    else:
        target.parent.mkdir(parents=True,exist_ok=True);archive=ROOT/'outputs/downloads'/(a.component+'-'+item['source_commit']+'.zip');archive.parent.mkdir(parents=True,exist_ok=True)
        urllib.request.urlretrieve(item['source_archive'],archive)
        with zipfile.ZipFile(archive) as z:
            prefix=z.namelist()[0].split('/')[0]+'/'
            for info in z.infolist():
                name=info.filename.removeprefix(prefix)
                if not name:continue
                dest=(target/name).resolve()
                if not dest.is_relative_to(target.resolve()):raise ValueError('Archive traversal rejected')
                if info.is_dir():dest.mkdir(parents=True,exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(info) as f,dest.open('wb') as out:shutil.copyfileobj(f,out)
        save(target/'zip2cad_source_receipt.json',dict(commit=item['source_commit'],url=item['source_archive'],archive_sha256=sha(archive)))
    if a.weights:
        weight=ROOT/item['weight_file'];weight.parent.mkdir(parents=True,exist_ok=True)
        if not weight.exists():
            partial=weight.with_suffix(weight.suffix+'.part');urllib.request.urlretrieve(item['weight_url'],partial)
            if sha(partial)!=item['weight_sha256']:raise ValueError('Weight SHA differs; partial download retained for diagnosis')
            partial.replace(weight)
        if sha(weight)!=item['weight_sha256']:raise ValueError('Existing weight hash differs; not overwritten')
        print('Weight verified:',weight)
    print('Install upstream requirements in the intended inference environment; see docs/INSTALL.md.')


if __name__=='__main__':main()
