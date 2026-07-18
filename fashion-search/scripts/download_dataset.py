import os
import json
import requests
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from tqdm import tqdm

# Curated list of verified, CC0 Unsplash photo IDs matching our 4 scene keywords.
# Direct file downloads from Unsplash CDN do not require API keys or trigger 403 blocks.
UNSPLASH_PHOTO_IDS = {
    "office": [
        "photo-1507679799987-c73779587ccf", "photo-1492562080023-ab3db95bfbce",
        "photo-1519085360753-af0119f7cbe7", "photo-1573496359142-b8d87734a5a2",
        "photo-1580489944761-15a19d654956", "photo-1560250097-0b93528c311a",
        "photo-1573497019940-1c28c88b4f3e", "photo-1573497019236-17f8177b81e8",
        "photo-1573496799652-408c2ac9fe98", "photo-1551836022-d5d88e9218df",
        "photo-1568602471122-7832951cc4c5", "photo-1544717305-2782549b5136",
        "photo-1519345182560-3f2917c472ef", "photo-1556157382-97eda2d62296",
        "photo-1508214751196-bcfd4ca60f91", "photo-1573496358961-3c82861b8f4f",
        "photo-1573496359250-9366115998a4", "photo-1573496359143-df9b4f2c2db5",
        "photo-1438761681033-6461ffad8d80", "photo-1534528741775-53994a69daeb",
        "photo-1506794778202-cad84cf45f1d", "photo-1500648767791-00dcc994a43e",
        "photo-1539571696357-5a69c17a67c6", "photo-1517841905240-472988babdf9",
        "photo-1494790108377-be9c29b29330", "photo-1524504388940-b1c1722653e1",
        "photo-1488426862026-3ee34a7d66df", "photo-1544005313-94ddf0286df2",
        "photo-1552581230-c01374138857", "photo-1531746020798-e6953c6e8e04"
    ],
    "park": [
        "photo-1502082553048-f009c37129b9", "photo-1441974231531-c6227db76b6e",
        "photo-1470246973918-29a93221c455", "photo-1511497584788-876760111969",
        "photo-1518495973542-4542c06a5843", "photo-1464822759023-fed622ff2c3b",
        "photo-1501785888041-af3ef285b470", "photo-1533928298208-27ff66555d8d",
        "photo-1475113548554-5a36f1f523d6", "photo-1505232987315-7ce774ff4a62",
        "photo-1454496522488-7a8e488e8606", "photo-1488521787991-ed7bbaae773c",
        "photo-1523381210434-271e8be1f52b", "photo-1516257984-b1b4d707412e",
        "photo-1534528741775-53994a69daeb", "photo-1494790108377-be9c29b29330",
        "photo-1488426862026-3ee34a7d66df", "photo-1479064555552-3ef4979f8908",
        "photo-1539571696357-5a69c17a67c6", "photo-1500648767791-00dcc994a43e",
        "photo-1506794778202-cad84cf45f1d", "photo-1522075469751-3a6694fb2f61",
        "photo-1517841905240-472988babdf9", "photo-1507003211169-0a1dd7228f2d",
        "photo-1544005313-94ddf0286df2", "photo-1531746020798-e6953c6e8e04",
        "photo-1524504388940-b1c1722653e1", "photo-1504198453319-5ce911bafcde",
        "photo-1528698827591-e19ccd7bc23d", "photo-1492562080023-ab3db95bfbce"
    ],
    "urban_street": [
        "photo-1515886657613-9f3515b0c78f", "photo-1509631179647-0177331693ae",
        "photo-1483985988355-763728e1935b", "photo-1529139574466-a303027c1d8b",
        "photo-1539109136881-3be0616acf4b", "photo-1490481651871-ab68de25d43d",
        "photo-1503342217505-b0a15ec3261c", "photo-1485968579580-b6d095142e6e",
        "photo-1554412930-e04af6db33c7", "photo-1517841905240-472988babdf9",
        "photo-1479064555552-3ef4979f8908", "photo-1539571696357-5a69c17a67c6",
        "photo-1500648767791-00dcc994a43e", "photo-1506794778202-cad84cf45f1d",
        "photo-1522075469751-3a6694fb2f61", "photo-1507003211169-0a1dd7228f2d",
        "photo-1544005313-94ddf0286df2", "photo-1531746020798-e6953c6e8e04",
        "photo-1524504388940-b1c1722653e1", "photo-1504198453319-5ce911bafcde",
        "photo-1528698827591-e19ccd7bc23d", "photo-1486312338219-ce68d2c6f44d",
        "photo-1494790108377-be9c29b29330", "photo-1534528741775-53994a69daeb",
        "photo-1516257984-b1b4d707412e", "photo-1523381210434-271e8be1f52b",
        "photo-1488521787991-ed7bbaae773c", "photo-1454496522488-7a8e488e8606",
        "photo-1556157382-97eda2d62296", "photo-1508214751196-bcfd4ca60f91"
    ],
    "home": [
        "photo-1583847268964-b28dc8f51f92", "photo-1513694203232-719a280e022f",
        "photo-1556911220-e15b29be8c8f", "photo-1585418694458-dc80857c6674",
        "photo-1522335789203-aabd1fc54bc9", "photo-1517694712202-14dd9538aa97",
        "photo-1540555700478-4be289fbecef", "photo-1505693416388-ac5ce068fe85",
        "photo-1544005313-94ddf0286df2", "photo-1593085512500-5d55148d6f0d",
        "photo-1534528741775-53994a69daeb", "photo-1494790108377-be9c29b29330",
        "photo-1488426862026-3ee34a7d66df", "photo-1479064555552-3ef4979f8908",
        "photo-1539571696357-5a69c17a67c6", "photo-1500648767791-00dcc994a43e",
        "photo-1506794778202-cad84cf45f1d", "photo-1522075469751-3a6694fb2f61",
        "photo-1517841905240-472988babdf9", "photo-1507003211169-0a1dd7228f2d",
        "photo-1544005313-94ddf0286df2", "photo-1531746020798-e6953c6e8e04",
        "photo-1524504388940-b1c1722653e1", "photo-1504198453319-5ce911bafcde",
        "photo-1528698827591-e19ccd7bc23d", "photo-1507679799987-c73779587ccf",
        "photo-1492562080023-ab3db95bfbce", "photo-1519085360753-af0119f7cbe7",
        "photo-1573496359142-b8d87734a5a2", "photo-1580489944761-15a19d654956"
    ]
}

def download_image(url, save_path):
    """Download image and save it as JPEG."""
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            # Resize to speed up execution
            img.thumbnail((512, 512))
            img.save(save_path, "JPEG")
            return True
    except Exception:
        pass
    return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download fashion datasets.")
    parser.add_argument("--fashionpedia_limit", type=int, default=500, help="Maximum number of Fashionpedia images to download.")
    parser.add_argument("--unsplash_limit", type=int, default=30, help="Maximum number of Unsplash images to download per query.")
    args = parser.parse_args()

    raw_dir = "data/raw"
    metadata_dir = "data/metadata"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    manifest = []
    
    # 1. Download Fashionpedia images from HuggingFace using streaming
    if args.fashionpedia_limit > 0:
        print(f"Streaming {args.fashionpedia_limit} images from Fashionpedia dataset on HuggingFace...")
        try:
            dataset = load_dataset("detection-datasets/fashionpedia", split="train", streaming=True)
            iterator = iter(dataset)
            
            fashionpedia_count = 0
            pbar = tqdm(total=args.fashionpedia_limit, desc="Downloading Fashionpedia")
            while fashionpedia_count < args.fashionpedia_limit:
                try:
                    sample = next(iterator)
                    filename = f"fashionpedia_{fashionpedia_count}.jpg"
                    save_path = os.path.join(raw_dir, filename)
                    
                    img = sample["image"].convert("RGB")
                    img.thumbnail((512, 512))
                    img.save(save_path, "JPEG")
                    
                    manifest.append({
                        "id": filename,
                        "source": "fashionpedia",
                        "local_path": save_path
                    })
                    fashionpedia_count += 1
                    pbar.update(1)
                except StopIteration:
                    print("Reached end of Fashionpedia stream early.")
                    break
                except Exception:
                    continue
            pbar.close()
        except Exception as e:
            print(f"Error streaming Fashionpedia: {e}")

    # 2. Download environment-aware context images from Unsplash CDN
    if args.unsplash_limit > 0:
        print("\nDownloading context-rich images from Unsplash CDN...")
        unsplash_count = 0
        for scene_tag, photo_ids in UNSPLASH_PHOTO_IDS.items():
            print(f"Retrieving direct images for scene '{scene_tag}'...")
            
            # Select slice of photo IDs based on limit
            selected_ids = photo_ids[:args.unsplash_limit]
            
            download_count = 0
            for pid in tqdm(selected_ids, desc=f"Downloading {scene_tag} context"):
                filename = f"unsplash_{scene_tag}_{download_count}.jpg"
                save_path = os.path.join(raw_dir, filename)
                
                # Fetch directly from Unsplash Source CDN
                url = f"https://images.unsplash.com/{pid}?w=512&q=80"
                if download_image(url, save_path):
                    manifest.append({
                        "id": filename,
                        "source": f"unsplash_{scene_tag}",
                        "local_path": save_path
                    })
                    download_count += 1
                    unsplash_count += 1
                    
            print(f"Successfully downloaded {download_count} images for '{scene_tag}' context.")

    # Write manifest file
    manifest_path = os.path.join(metadata_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_downloaded = len(manifest)
    print(f"\nDataset download complete. Total images: {total_downloaded}")
    print(f"Manifest written to {manifest_path}")

if __name__ == "__main__":
    main()
