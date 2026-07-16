import os
import sys
import zipfile
import shutil
import urllib.request

def main():
    if len(sys.argv) < 3:
        sys.exit(1)
        
    src = sys.argv[1]
    dest = sys.argv[2]
    
    os.makedirs(dest, exist_ok=True)
    temp_zip = os.path.join(dest, "temp.zip")
    print(f"Downloading and extracting {src} to {dest}...")
    try:
        if src.startswith(("http://", "https://")):
            req = urllib.request.Request(
                src, 
                headers={'User-Agent': 'Mozilla/5.0'} # Helps bypass some bot blocks
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(temp_zip, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        else:
            shutil.copy2(src, temp_zip)
        print(f"Extracting {temp_zip} to {dest}...")
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            for member in zip_ref.infolist():
                if member.is_dir():
                    continue
                member.filename = os.path.basename(member.filename)
                zip_ref.extract(member, dest)
    finally:
        print(f"Cleaning up temporary files...")
        if os.path.exists(temp_zip):
            os.remove(temp_zip)

if __name__ == "__main__":
    main()
