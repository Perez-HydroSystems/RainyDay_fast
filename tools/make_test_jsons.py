"""
GP 2026-10-22: JSONs of the RainyDay_fast validation matrix (production settings of the 2026-10-21 campaign).

    python tools/make_test_jsons.py <variant> [--rainpath SET=DIR ...] [--catalog-from VARIANT] [--set KEY=VALUE ...]
                                    [--only NAME ...]

Writes $FAST_TEST_ROOT/jsons/<variant>/<name>.json (default root ~/Databases/RainyDay_fast_tests) for:
  Talequah    Talequah                          (smoke test: 2002 Jul-Sep, 10 storms, NYEARS 100, 5 realizations)
  Iowa        Pixel_r1_A1km2_rot0               (1 cell)
  Iowa        Circular_r1_A16000km2_rot0        (largest basin)
  Iowa        Elliptical_r10_A16000km2_rot45    (bounding box much larger than the mask)
  HUC_Texas   Basin_raw_Texas_HUC_120401020101  (Texas domain, HUC basin)
Templates: the campaign JSONs in $FAST_KIT_JSON (a folder of production JSONs; required for Iowa and HUC_Texas) and
Examples/Talequah/Talequah.json (shapefiles taken from this repo).
  MAINPATH      -> $FAST_TEST_ROOT/runs/<variant>/<set>
  --rainpath    -> RAINPATH = DIR/AORC.*.preciptemp.nc for that set (default: the template's data_aorcfix2)
  --catalog-from-> CREATECATALOG false, CATALOGPATH = the StormCatalog of that variant's run (frequency analysis only)
  --set         -> any other key, value parsed as JSON when possible (e.g. --set MULTIDURATIONENGINE='"table"')
"""
import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("FAST_TEST_ROOT", str(Path.home() / "Databases" / "RainyDay_fast_tests")))
KIT = Path(os.environ.get("FAST_KIT_JSON", str(Path.home() / "RainyDay_fast_kit_json")))
SHAPES = [("Talequah", "Talequah"),
          ("Iowa", "Pixel_r1_A1km2_rot0"),
          ("Iowa", "Circular_r1_A16000km2_rot0"),
          ("Iowa", "Elliptical_r10_A16000km2_rot45"),
          ("HUC_Texas", "Basin_raw_Texas_HUC_120401020101")]
DURS = [1, 3, 6, 12, 24, 48, 72]


def talequah():
    b = json.load(open(REPO / "Examples" / "Talequah" / "Talequah.json"))
    shp = REPO / "Examples" / "Talequah" / "Shapefiles"
    b.update(SCENARIONAME="Talequah", CATALOGNAME="Talequah.nc",
             DOMAINSHP=str(shp / "Talequah_transposition_domain.shp"), WATERSHEDSHP=str(shp / "Talequah_basin.shp"),
             RAINPATH=str(Path.home() / "Databases" / "AORC_RainyDay" / "data_aorcfix2" / "AORC.*.preciptemp.nc"),
             FREQANALYSIS="true", DIAGNOSTICPLOTS="only-statistics", NYEARS=100, NREALIZATIONS=5,
             UNCERTAINTY="all_realizations", RETURNLEVELS=[2, 5, 10, 25, 50, 100], SCENARIOS="false",
             DURATIONCORRECTION="false", DURATION=72, MULTIPLEDURATIONS=DURS, MULTIDURATIONENGINE="fast")
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--rainpath", nargs="*", default=[])
    ap.add_argument("--catalog-from")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    rain = dict(x.split("=", 1) for x in a.rainpath)
    extra = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            extra[k] = json.loads(v)
        except ValueError:
            extra[k] = v
    out = ROOT / "jsons" / a.variant
    out.mkdir(parents=True, exist_ok=True)
    for st, name in SHAPES:
        if a.only and name not in a.only:
            continue
        j = talequah() if st == "Talequah" else json.load(open(KIT / st / f"{name}.json"))
        j["MAINPATH"] = str(ROOT / "runs" / a.variant / st)
        Path(j["MAINPATH"]).mkdir(parents=True, exist_ok=True)              # RainyDay does os.mkdir below it
        if st in rain:
            j["RAINPATH"] = str(Path(rain[st]) / "AORC.*.preciptemp.nc")
        if a.catalog_from:
            j["CREATECATALOG"] = "false"
            j["CATALOGPATH"] = str(ROOT / "runs" / a.catalog_from / st / f"{name}_72" / "StormCatalog" / j["CATALOGNAME"])
        j.update(extra)
        for k in ("DOMAINSHP", "WATERSHEDSHP"):
            assert Path(j[k]).exists(), f"{name}: missing {j[k]}"
        json.dump(j, open(out / f"{name}.json", "w"), indent=4)
        print(f"{out / (name + '.json')}  RAINPATH={j['RAINPATH']}"
              + (f"  CATALOGPATH={j['CATALOGPATH']}" if a.catalog_from else ""))


if __name__ == "__main__":
    main()
