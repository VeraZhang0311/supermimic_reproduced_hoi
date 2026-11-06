"""Create a frozen ULTRA train / Protected eval manifest without loading motions."""
import argparse
import json

from .data import prepare_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-inputs", required=True, help="existing hoi_v1_frozen_inputs_20261023.tar.gz")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--source-root", help="explicit remap of canonical source root; original identities preserved")
    args = parser.parse_args()
    manifest = prepare_manifest(args.frozen_inputs, args.output, source_root=args.source_root)
    print(json.dumps({"content_sha256": manifest["content_sha256"], "partitions": manifest["partitions"],
                      "protected_eval_motions": sum(s["motions"] for p, s in manifest["partitions"].items() if p != "train")}, indent=2))


if __name__ == "__main__":
    main()
