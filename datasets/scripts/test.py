import datasets

def get_audio_durations_and_labels(ds_split):
    durations = []
    labels = []

    short_sub500_indices = []
    short_sub500_labels = []

    long_1min_indices = []
    long_1min_labels = []

    long_30_60_indices = []
    long_30_60_labels = []

    for idx, example in enumerate(ds_split):
        audio = example["audio"]
        label = example["label"]
        duration_sec = len(audio["array"]) / audio["sampling_rate"]

        durations.append(duration_sec)
        labels.append(label)

        if duration_sec < 0.5:
            short_sub500_indices.append(idx)
            short_sub500_labels.append(label)
        elif duration_sec > 60:
            long_1min_indices.append(idx)
            long_1min_labels.append(label)
        elif duration_sec > 30:
            long_30_60_indices.append(idx)
            long_30_60_labels.append(label)

    return (
        durations, labels,
        short_sub500_indices, short_sub500_labels,
        long_1min_indices, long_1min_labels,
        long_30_60_indices, long_30_60_labels
    )

def bucket_durations(durations):
    bins = {
        "<0.5 sec": [],
        "0-5 sec": [],
        "5-10 sec": [],
        "10-20 sec": [],
        "20-30 sec": [],
        "30-60 sec": [],
        "60+ sec": []
    }

    for d in durations:
        if d < 0.5:
            bins["<0.5 sec"].append(d)
        elif d <= 5:
            bins["0-5 sec"].append(d)
        elif d <= 10:
            bins["5-10 sec"].append(d)
        elif d <= 20:
            bins["10-20 sec"].append(d)
        elif d <= 30:
            bins["20-30 sec"].append(d)
        elif d <= 60:
            bins["30-60 sec"].append(d)
        else:
            bins["60+ sec"].append(d)

    return bins

def _main():
    print("📦 Loading dataset...")
    ds = datasets.load_dataset("vtsouval/flusense", trust_remote_code=True)["train"]

    print("🔍 Analyzing durations...")
    (
        durations, labels,
        sub500_indices, sub500_labels,
        over_60s, over_60s_labels,
        between_30_60s, between_30_60s_labels
    ) = get_audio_durations_and_labels(ds)

    print("\n🎧 Audio Length Stats:")
    print(f"  • Total samples: {len(durations)}")
    print(f"  • Min duration: {min(durations):.3f} sec")
    print(f"  • Max duration: {max(durations):.2f} sec")
    print(f"  • Avg duration: {sum(durations)/len(durations):.2f} sec")

    print("\n📊 Distribution by duration bins (with average durations):")
    bin_durations = bucket_durations(durations)
    for label, durs in bin_durations.items():
        count = len(durs)
        avg = sum(durs) / count if count > 0 else 0.0
        pct = 100 * count / len(durations)
        print(f"  • {label:<10}: {count:5} samples ({pct:.2f}%) — Avg: {avg:.2f} sec")

    print("\n🧾 Short sample stats (<0.5 sec):")
    print(f"  • Count: {len(sub500_indices)}")
    print(f"    Indices: {sub500_indices[:10]}{' ...' if len(sub500_indices) > 10 else ''}")
    print(f"    Labels : {sub500_labels[:10]}{' ...' if len(sub500_labels) > 10 else ''}")

    print("\n🧾 Long sample stats:")
    print(f"  • > 60 sec: {len(over_60s)}")
    print(f"    Indices: {over_60s[:10]}{' ...' if len(over_60s) > 10 else ''}")
    print(f"    Labels : {over_60s_labels[:10]}{' ...' if len(over_60s_labels) > 10 else ''}")

    print(f"\n  • 30–60 sec: {len(between_30_60s)}")
    print(f"    Indices: {between_30_60s[:10]}{' ...' if len(between_30_60s) > 10 else ''}")
    print(f"    Labels : {between_30_60s_labels[:10]}{' ...' if len(between_30_60s_labels) > 10 else ''}")

    # Save all flagged indices to file
    with open("short_sub500_indices.txt", "w") as f:
        f.write("\n".join(map(str, sub500_indices)))

    with open("long_60plus_indices.txt", "w") as f:
        f.write("\n".join(map(str, over_60s)))

    with open("long_30to60_indices.txt", "w") as f:
        f.write("\n".join(map(str, between_30_60s)))

import datasets
from collections import Counter, defaultdict

EXCLUDED_LABELS = {"etc", "vomit", "snore", "wheeze"}
INDEX_FILES = [
    # "short_sub500_indices.txt",
    "long_30to60_indices.txt",
    "long_60plus_indices.txt"
]

DURATION_BINS = [
    ("<0.5 sec", lambda d: d < 0.5),
    ("0-5 sec", lambda d: 0.5 <= d <= 5),
    ("5-10 sec", lambda d: 5 < d <= 10),
    ("10-20 sec", lambda d: 10 < d <= 20),
    ("20-30 sec", lambda d: 20 < d <= 30),
    ("30-60 sec", lambda d: 30 < d <= 60),
    ("60+ sec", lambda d: d > 60),
]

def load_excluded_indices():
    indices = set()
    for path in INDEX_FILES:
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.isdigit():
                        indices.add(int(line))
        except FileNotFoundError:
            print(f"⚠️ File not found: {path}")
    return indices

def bucket_durations(ds):
    bin_counts = defaultdict(list)  # bin_name -> list of durations
    bin_labels = defaultdict(set)   # bin_name -> set of labels

    for example in ds:
        duration = len(example["audio"]["array"]) / example["audio"]["sampling_rate"]
        label = example["label"]

        for bin_name, condition in DURATION_BINS:
            if condition(duration):
                bin_counts[bin_name].append(duration)
                bin_labels[bin_name].add(label)
                break

    return bin_counts, bin_labels

def main():
    print("📦 Loading dataset...")
    ds = datasets.load_dataset("CSTR-Edinburgh/vctk", trust_remote_code=True)["train"]

    print("🚫 Loading indices of samples to exclude...")
    excluded_indices = load_excluded_indices()
    print(f"  • Total indices to exclude: {len(excluded_indices)}")

    print("🧹 Filtering dataset...")

    # Step 1: remove based on index
    ds = ds.select([i for i in range(len(ds)) if i not in excluded_indices])

    # Step 2: remove based on labels
    ds = ds.filter(lambda x: x["label"] not in EXCLUDED_LABELS)

    print(f"✅ Remaining samples after filtering: {len(ds)}")

    # Step 3: label distribution
    label_counts = Counter(ds["label"])
    total = sum(label_counts.values())

    print("\n📊 Label distribution after filtering:")
    print(f"{'Label':<20} {'Count':<10} {'Percent':<10}")
    print("-" * 45)
    for label, count in label_counts.most_common():
        pct = 100 * count / total
        print(f"{label:<20} {count:<10} {pct:.2f}%")

    # Step 4: duration binning
    print("\n⏱️ Duration bin analysis:")
    bin_counts, bin_labels = bucket_durations(ds)

    for bin_name in DURATION_BINS:
        label = bin_name[0]
        durations = bin_counts[label]
        if durations:
            avg = sum(durations) / len(durations)
            print(f"  • {label:<10}: {len(durations):5} samples — Avg: {avg:.2f} sec")
            print(f"     Labels: {sorted(bin_labels[label])}")
        else:
            print(f"  • {label:<10}:     0 samples")

if __name__ == "__main__":
    ds = datasets.load_dataset("CSTR-Edinburgh/vctk", trust_remote_code=True)
    print(ds)
    #main()