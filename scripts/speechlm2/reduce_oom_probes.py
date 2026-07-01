#!/usr/bin/env python
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reduce distributed_oomptimizer.py probe results into a ready-to-paste bucketing profile.

The distributed OOMptimizer already prints the final profile at the end of a *successful* run. This
standalone helper is for the case where a run was killed midway (the tool prints nothing then): it reads
whatever per-probe ``probe_*.jsonl`` records exist on disk and reconstructs a partial profile.

It scans the given directory recursively, so pointing it at the parent of several job-unique probe
directories merges them (keep only runs with identical settings -- memory fraction, buckets, etc. -- or
the merge mixes incomparable batch sizes).

Usage::

    python scripts/speechlm2/reduce_oom_probes.py /path/to/oomptimizer_probes_dir
"""

import argparse
import glob
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probe_dir", help="Directory scanned recursively for probe_*.jsonl files.")
    args = parser.parse_args()

    best: dict[float, int] = {}  # bucket -> largest usable batch size
    for path in glob.glob(os.path.join(args.probe_dir, "**", "probe_*.jsonl"), recursive=True):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    bucket = float(rec["bucket"])
                    bs = int(rec["batch_size"])
                except (ValueError, TypeError, KeyError):
                    continue  # skip malformed / 2D-bucket lines
                if rec["status"] in ("ok", "memory_target"):
                    best[bucket] = max(best.get(bucket, 0), bs)
                else:  # oom / other -> make sure the bucket at least shows up as 0
                    best.setdefault(bucket, 0)

    if not best:
        raise SystemExit(f"No probe records found in {args.probe_dir}")

    def fmt(b: float):  # keep integral buckets as ints
        return int(b) if float(b).is_integer() else b

    # Enforce a monotonically non-increasing batch size over ascending buckets. Independent per-bucket
    # searches can leak an over-estimate where a larger bucket ends up with a bigger batch size than a
    # smaller one, which then OOMs in real training. Clamp each to the previous (smaller) bucket's size.
    prev_bs = None
    for bucket in sorted(best):
        if prev_bs is not None and best[bucket] > prev_bs:
            best[bucket] = prev_bs
        prev_bs = best[bucket]

    # ascending by bucket, then merge runs of identical batch sizes (keep the largest bin)
    merged: list[list] = []  # [bin, batch_size]
    for bucket in sorted(best):
        bs = best[bucket]
        if merged and merged[-1][1] == bs:
            merged[-1][0] = bucket
        else:
            merged.append([bucket, bs])

    bins = [fmt(b) for b, _ in merged]
    sizes = [bs for _, bs in merged]

    print(f"num_buckets: {len(bins)}")
    print("bucket_duration_bins: [" + ",".join(map(str, bins)) + "]")
    print("bucket_batch_size: [" + ",".join(map(str, sizes)) + "]")


if __name__ == "__main__":
    main()
