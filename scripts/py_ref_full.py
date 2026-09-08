#!/usr/bin/env python3
"""Decode a whole .s16 file with the reference Python ld-decode, mirroring the
Rust port's serial loop, and print per-field JSON metadata for comparison."""

import sys
import json
import logging

sys.path.insert(0, "../_ld-decode-ref")

import numpy as np
import lddecode.core as core_mod
from lddecode.core import RFDecode, Field, DemodCache
from lddecode.utils import make_loader

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("refdecode")
core_mod.logger = logger


class NumpyLoader:
    def __init__(self, data):
        self.data = data

    def __call__(self, infile, offset, blocklen):
        if offset >= len(self.data):
            return None
        return self.data[offset : offset + blocklen].copy()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "test.s16"
    data = np.fromfile(path, dtype="<i2")
    print(f"loaded {len(data)} samples from {path}", file=sys.stderr)

    rf = RFDecode(inputfreq=40, system="NTSC")
    blocksize = rf.blocklen - (rf.blockcut + rf.blockcut_end)
    readlen = rf.linelen * 400

    dc = DemodCache(
        rf, None, NumpyLoader(data), {}, num_worker_threads=0, cachesize=128
    )

    fdoffset = 0.0
    prevfield = None
    fields_written = 0
    out_fields = []

    # Capture meanlinelen and line0loc for comparison.
    orig_computeLineLen = Field.computeLineLen
    orig_getLine0 = Field.getLine0

    def computeLineLen_wrap(self, validpulses):
        self.meanlinelen_debug = orig_computeLineLen(self, validpulses)
        return self.meanlinelen_debug

    def getLine0_wrap(self, validpulses, meanlinelen):
        rv = orig_getLine0(self, validpulses, meanlinelen)
        self.line0loc_debug = rv[0]
        return rv

    Field.computeLineLen = computeLineLen_wrap
    Field.getLine0 = getLine0_wrap

    max_iter = 400
    while fields_written < 40 and max_iter > 0:
        max_iter -= 1
        start = int(fdoffset)
        readloc = max(int(start - rf.blockcut), 0)
        readloc_block = readloc // rf.blocklen
        numblocks = (readlen // rf.blocklen) + 2

        decode = dc.read(readloc_block * rf.blocklen, numblocks * rf.blocklen)
        if decode is None:
            print("EOF", file=sys.stderr)
            break


        f = Field(
            rf,
            decode,
            prevfield=prevfield,
            initphase=False,
            fields_written=fields_written,
            readloc=decode["startloc"],
        )
        # Debug: raw/valid pulses before process() overrides them.
        raw = f.getpulses()
        f.rawpulses = raw
        vp = f.refinepulses()
        f.validpulses = vp
        print(
            f"PYDBG fieldstart={decode['startloc']} raw={len(raw)} valid={len(vp)} "
            f"vp20={[(v[0], int(v[1].start), v[1].len) for v in vp[:20]]}",
            file=sys.stderr,
        )
        if fields_written == 0:
            from collections import Counter
            lh = Counter(int(p.len) for p in raw)
            print(
                "PYDBG rawlens=" + str(sorted(lh.items())),
                file=sys.stderr,
            )
            vplh = Counter((int(v[0]), int(v[1].len)) for v in vp)
            print(
                "PYDBG vplens=" + str(sorted(vplh.items())),
                file=sys.stderr,
            )
            # print all non-hsync validpulses with positions
            nz = [(int(v[0]), int(v[1].start), int(v[1].len)) for v in vp if v[0] != 0]
            print("PYDBG vp_nonzero=" + str(nz), file=sys.stderr)
        f.process()

        offset = (
            f.nextfieldoffset - (readloc - decode["startloc"]) if f.valid else f.nextfieldoffset
        )
        if f.valid:
            offset = f.nextfieldoffset - (readloc - decode["startloc"])
            fi = {
                "isFirstField": f.isFirstField,
                "syncConf": f.sync_confidence,
                "seqNo": fields_written + 1,
                "diskLoc": round(f.readloc / (rf.freq_hz / (rf.SysParams["FPS"] * 2)), 1),
                "fileLoc": f.readloc,
                "medianBurstIre": getattr(f, "medianBurstIre", None),
                "vblank_next": getattr(f, "vblank_next", None),
                "meanlinelen": getattr(f, "meanlinelen_debug", None),
                "line0loc": getattr(f, "line0loc_debug", None),
                "linelocs_first5": [round(x, 1) for x in f.linelocs[:5]] if len(f.linelocs) else None,
                "fieldPhaseID": getattr(f, "fieldPhaseID", None),
                "decodeFaults": getattr(f, "decodeFaults", None),
                "linecount": f.linecount,
                "outlinelen": f.outlinelen,
                "linelocs_first": f.linelocs[0] if len(f.linelocs) else None,
                "linelocs_last": f.linelocs[-1] if len(f.linelocs) else None,
                "nextfieldoffset": f.nextfieldoffset,
            }
            out_fields.append(fi)
            fields_written += 1
            prevfield = f
            def _def(o):
                if isinstance(o, np.integer):
                    return int(o)
                if isinstance(o, np.floating):
                    return float(o)
                return str(o)
            print(json.dumps(fi, default=_def), file=sys.stderr)
        else:
            prevfield = None

        fdoffset = fdoffset + offset

    dc.end()

    def _def(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return str(o)

    print(json.dumps(out_fields, default=_def))


if __name__ == "__main__":
    main()
