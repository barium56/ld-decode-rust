#!/usr/bin/env python3
"""Drive the reference Python ld-decode on a synthetic signal to compare sync
detection against the Rust port.

Usage: python py_ref_decode.py test.s16
"""

import sys
import logging

sys.path.insert(0, "../_ld-decode-ref")

import numpy as np
import lddecode.core as core_mod
from lddecode.core import RFDecode, Field, DemodCache

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("refdecode")
core_mod.logger = logger


class NumpyLoader:
    """loader with the (infile, offset, blocklen) signature the DemodCache expects."""

    def __init__(self, data):
        self.data = data

    def __call__(self, infile, offset, blocklen):
        if offset >= len(self.data):
            return None
        return self.data[offset : offset + blocklen].copy()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "test.s16"
    data = np.fromfile(path, dtype="<i2")
    print(f"loaded {len(data)} samples from {path}")

    rf = RFDecode(inputfreq=40, system="NTSC")
    blocksize = rf.blocklen - (rf.blockcut + rf.blockcut_end)
    readlen = rf.linelen * 400
    print(
        f"linelen={rf.linelen} blocksize={blocksize} readlen={readlen} "
        f"blockcut={rf.blockcut} blockcut_end={rf.blockcut_end}"
    )

    dc = DemodCache(
        rf, None, NumpyLoader(data), {}, num_worker_threads=0, cachesize=64
    )

    # Decode a chunk covering roughly two fields from the start.
    begin = 0
    for attempt in range(3):
        decode = dc.read(begin, readlen)
        if decode is None:
            print("EOF while demodulating")
            break
        print(
            f"decode startloc={decode['startloc']} input={None if decode['input'] is None else len(decode['input'])} "
            f"video={None if decode['video'] is None else len(decode['video'])} "
            f"rfhpf={None if decode['rfhpf'] is None else len(decode['rfhpf'])}"
        )
        if decode["video"] is None or len(decode["video"]) < 1000:
            print("no video data")
            break

        f = Field(rf, decode, fields_written=0, readloc=decode["startloc"])
        f.process()

        print(
            f"attempt={attempt} valid={f.valid} linelocs1={None if f.linelocs1 is None else len(f.linelocs1)} "
            f"linebad={None if f.linebad is None else len(f.linebad)} "
            f"isFirstField={f.isFirstField} nextfieldoffset={f.nextfieldoffset} "
            f"sync_confidence={f.sync_confidence}"
        )
        if f.linelocs1 is not None:
            print(f"first 5 linelocs: {f.linelocs1[:5]}")
            print(f"last 5 linelocs: {f.linelocs1[-5:]}")
            print(f"line0 candidates ok")
            break
        # move on to the next chunk
        begin += readlen // 2

    dc.end()


if __name__ == "__main__":
    main()
