"""Create the GitHub source archive from git's own ignore-aware file selection."""
import argparse,json,subprocess,zipfile,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'dist/Zip2CAD1.0-source.zip');a=p.parse_args()
    files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    files=sorted(set(n for n in files if n));a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED) as z:
        manifest={}
        for name in files:
            file=ROOT/name
            if file.stat().st_size>10_000_000:raise ValueError('Unexpected large public source: '+name)
            z.write(file,'Zip2CAD1.0/'+name)
            manifest[name]=hashlib.sha256(file.read_bytes()).hexdigest()
        z.writestr('Zip2CAD1.0/SOURCE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(output=str(a.output),files=len(files),MB=a.output.stat().st_size/1e6),indent=2))


if __name__=='__main__':main()
