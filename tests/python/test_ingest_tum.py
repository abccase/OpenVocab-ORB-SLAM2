from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from tools.ingest_tum import (
    EXPECTED_ARCHIVES,
    DatasetSpec,
    UnsafeArchive,
    associate_streams,
    build_sequence_manifest,
    inspect_archive,
    ingest_datasets,
    parse_tum_index,
    safe_extract_archive,
    validate_inbox,
)

import cv2
import numpy as np


def write_tar(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for info, payload in members:
            if info.isreg():
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            else:
                archive.addfile(info)


class ArchiveSafetyTests(unittest.TestCase):
    def test_inbox_requires_exact_six_archive_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inbox = Path(temporary)
            (inbox / EXPECTED_ARCHIVES[0]).write_bytes(b"one")
            (inbox / "unexpected.tgz").write_bytes(b"two")

            result = validate_inbox(inbox)

            self.assertEqual(result.missing, tuple(EXPECTED_ARCHIVES[1:]))
            self.assertEqual(result.unexpected, ("unexpected.tgz",))
            self.assertFalse(result.ok)

    def test_rejects_absolute_parent_and_wrong_top_level_members(self) -> None:
        cases = ("/etc/passwd", "../../escape", "safe/../../escape", "other/file.txt")
        for member in cases:
            with self.subTest(member=member), tempfile.TemporaryDirectory() as temporary:
                archive = Path(temporary) / "unsafe.tgz"
                write_tar(archive, [(tarfile.TarInfo(member), b"bad")])

                with self.assertRaises(UnsafeArchive):
                    inspect_archive(archive, "safe")

    def test_rejects_links_devices_and_duplicate_normalized_paths(self) -> None:
        unsafe_members: list[tarfile.TarInfo] = []
        symbolic = tarfile.TarInfo("safe/link")
        symbolic.type = tarfile.SYMTYPE
        symbolic.linkname = "../../escape"
        unsafe_members.append(symbolic)
        hard = tarfile.TarInfo("safe/hard")
        hard.type = tarfile.LNKTYPE
        hard.linkname = "safe/file"
        unsafe_members.append(hard)
        device = tarfile.TarInfo("safe/device")
        device.type = tarfile.CHRTYPE
        unsafe_members.append(device)

        for member in unsafe_members:
            with self.subTest(kind=member.type), tempfile.TemporaryDirectory() as temporary:
                archive = Path(temporary) / "unsafe.tgz"
                write_tar(archive, [(member, b"")])
                with self.assertRaises(UnsafeArchive):
                    inspect_archive(archive, "safe")

        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "duplicate.tgz"
            write_tar(
                archive,
                [
                    (tarfile.TarInfo("safe/rgb/a.png"), b"a"),
                    (tarfile.TarInfo("safe/rgb/./a.png"), b"b"),
                ],
            )
            with self.assertRaises(UnsafeArchive):
                inspect_archive(archive, "safe")

    def test_safe_extraction_is_atomic_and_promotes_only_valid_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "valid.tgz"
            top = "rgbd_dataset_tiny"
            write_tar(
                archive,
                [
                    (tarfile.TarInfo(f"{top}/rgb.txt"), b"1.0 rgb/1.png\n"),
                    (tarfile.TarInfo(f"{top}/depth.txt"), b"1.0 depth/1.png\n"),
                    (tarfile.TarInfo(f"{top}/groundtruth.txt"), b"1.0 0 0 0 0 0 0 1\n"),
                ],
            )
            output = root / "raw"

            extracted = safe_extract_archive(archive, output, top)

            self.assertEqual(extracted, output / top)
            self.assertEqual((extracted / "rgb.txt").read_text(), "1.0 rgb/1.png\n")
            self.assertFalse(any(output.glob(".*.partial-*")))


class AssociationTests(unittest.TestCase):
    def test_association_is_one_to_one_stable_and_in_official_order(self) -> None:
        rgb = [(1.000, "rgb/a.png"), (1.010, "rgb/b.png"), (1.040, "rgb/c.png")]
        depth = [
            (0.990, "depth/early.png"),
            (1.010, "depth/exact.png"),
            (1.020, "depth/tie.png"),
            (1.060, "depth/boundary.png"),
        ]

        rows = associate_streams(rgb, depth, max_difference=0.02)

        self.assertEqual(
            rows,
            [
                (1.000, "rgb/a.png", 0.990, "depth/early.png"),
                (1.010, "rgb/b.png", 1.010, "depth/exact.png"),
                (1.040, "rgb/c.png", 1.020, "depth/tie.png"),
            ],
        )
        self.assertEqual(len({row[3] for row in rows}), len(rows))

    def test_association_rejects_beyond_boundary_and_duplicate_timestamps(self) -> None:
        self.assertEqual(
            associate_streams([(1.0, "rgb/a")], [(1.020001, "depth/a")], 0.02),
            [],
        )
        with self.assertRaises(ValueError):
            associate_streams(
                [(1.0, "rgb/a"), (1.0, "rgb/b")],
                [(1.0, "depth/a")],
                0.02,
            )


class DatasetContractTests(unittest.TestCase):
    def test_index_rejects_duplicate_timestamp_and_unsafe_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index = Path(temporary) / "rgb.txt"
            index.write_text("1.0 rgb/a.png\n1.0 rgb/b.png\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_tum_index(index, expected_columns=2)
            index.write_text("1.0 ../escape.png\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_tum_index(index, expected_columns=2)

    def test_manifest_validates_images_association_and_original_tree_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "rgbd_dataset_tiny"
            (root / "rgb").mkdir(parents=True)
            (root / "depth").mkdir()
            rgb_a = np.zeros((2, 3, 3), dtype=np.uint8)
            rgb_b = np.full((2, 3, 3), 127, dtype=np.uint8)
            depth_a = np.array([[0, 1000, 2000], [3000, 4000, 5000]], dtype=np.uint16)
            depth_b = np.array([[6000, 7000, 8000], [9000, 10000, 11000]], dtype=np.uint16)
            self.assertTrue(cv2.imwrite(str(root / "rgb/1.png"), rgb_a))
            self.assertTrue(cv2.imwrite(str(root / "rgb/2.png"), rgb_b))
            self.assertTrue(cv2.imwrite(str(root / "depth/1.png"), depth_a))
            self.assertTrue(cv2.imwrite(str(root / "depth/2.png"), depth_b))
            (root / "rgb.txt").write_text(
                "# RGB\n1.000 rgb/1.png\n2.000 rgb/2.png\n", encoding="utf-8"
            )
            (root / "depth.txt").write_text(
                "1.010 depth/1.png\n1.990 depth/2.png\n", encoding="utf-8"
            )
            (root / "groundtruth.txt").write_text(
                "1.000 0 0 0 0 0 0 1\n2.000 1 0 0 0 0 0 1\n", encoding="utf-8"
            )
            archive = Path(temporary) / "tiny.tgz"
            archive.write_bytes(b"archive identity")
            manifest_path = Path(temporary) / "manifest.json"

            first = build_sequence_manifest(
                sequence_id="tiny",
                dataset_root=root,
                archive_path=archive,
                settings="TUM1.yaml",
                manifest_path=manifest_path,
            )
            second = build_sequence_manifest(
                sequence_id="tiny",
                dataset_root=root,
                archive_path=archive,
                settings="TUM1.yaml",
                manifest_path=manifest_path,
            )

            self.assertEqual(first["validation_status"], "VALID")
            self.assertEqual(first["counts"], {"rgb": 2, "depth": 2, "groundtruth": 2, "associations": 2})
            self.assertEqual(first["image_dimensions"], {"width": 3, "height": 2})
            self.assertEqual(first["depth"], {"dtype": "uint16", "min_positive": 1000, "max": 11000})
            self.assertEqual(first["extracted_tree_sha256"], second["extracted_tree_sha256"])
            self.assertEqual(len(first["archive_sha256"]), 64)
            self.assertEqual(len(first["association_sha256"]), 64)
            self.assertEqual(json.loads(manifest_path.read_text()), second)
            self.assertEqual(
                (root / "associate.txt").read_text(encoding="utf-8").splitlines(),
                [
                    "1.000000000 rgb/1.png 1.010000000 depth/1.png",
                    "2.000000000 rgb/2.png 1.990000000 depth/2.png",
                ],
            )

    def test_ingest_datasets_extracts_and_resumes_validated_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source/rgbd_dataset_tiny"
            (source / "rgb").mkdir(parents=True)
            (source / "depth").mkdir()
            self.assertTrue(cv2.imwrite(str(source / "rgb/1.png"), np.zeros((2, 3, 3), np.uint8)))
            self.assertTrue(cv2.imwrite(str(source / "depth/1.png"), np.ones((2, 3), np.uint16)))
            (source / "rgb.txt").write_text("1.0 rgb/1.png\n", encoding="utf-8")
            (source / "depth.txt").write_text("1.0 depth/1.png\n", encoding="utf-8")
            (source / "groundtruth.txt").write_text("1.0 0 0 0 0 0 0 1\n", encoding="utf-8")
            inbox = base / "inbox"
            inbox.mkdir()
            archive = inbox / "rgbd_dataset_tiny.tgz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(source, arcname=source.name)
            raw = base / "raw"
            manifests = base / "manifests"
            spec = DatasetSpec("tiny", archive.name, "TUM1.yaml")

            first = ingest_datasets([spec], inbox, raw, manifests, required_free_bytes=1)
            second = ingest_datasets([spec], inbox, raw, manifests, required_free_bytes=1)

            self.assertEqual(first, second)
            self.assertEqual(first[0]["validation_status"], "VALID")
            self.assertTrue((raw / "rgbd_dataset_tiny/associate.txt").is_file())
            self.assertTrue((manifests / "tiny.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
