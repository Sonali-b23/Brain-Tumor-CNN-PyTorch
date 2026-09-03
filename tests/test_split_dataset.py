import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.split_dataset import dedupe_by_content, split_dataset


def _write_file(path, content):
    with open(path, 'wb') as f:
        f.write(content)


def test_dedupe_by_content_drops_byte_identical_duplicates(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"  # identical content to a, different name
    c = tmp_path / "c.jpg"  # different content

    _write_file(a, b"same-bytes")
    _write_file(b, b"same-bytes")
    _write_file(c, b"different-bytes")

    unique_files, dropped = dedupe_by_content([str(a), str(b), str(c)])

    assert len(unique_files) == 2
    assert len(dropped) == 1
    # the first-encountered file for a given content hash is kept
    assert str(a) in unique_files
    assert str(b) not in unique_files
    assert str(c) in unique_files


def test_dedupe_by_content_keeps_all_when_no_duplicates(tmp_path):
    files = []
    for i in range(5):
        p = tmp_path / f"img_{i}.jpg"
        _write_file(p, f"content-{i}".encode())
        files.append(str(p))

    unique_files, dropped = dedupe_by_content(files)

    assert len(unique_files) == 5
    assert dropped == []


def test_split_dataset_produces_no_cross_split_duplicates(tmp_path):
    """The regression test for the actual leakage bug: build a source
    dataset that intentionally contains duplicate-content images under
    different filenames (mirroring the real Kaggle dataset), split it, and
    assert that no image (by content hash) appears in more than one of
    train/val/test."""
    import hashlib

    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    for category in ["yes", "no"]:
        os.makedirs(source_dir / category, exist_ok=True)

    # 20 unique images per category, plus 5 duplicate copies (of existing
    # images, under new filenames) per category -- same shape as the real
    # dataset's duplication pattern.
    for category in ["yes", "no"]:
        for i in range(20):
            _write_file(source_dir / category / f"{category}_{i}.jpg",
                        f"{category}-content-{i}".encode())
        for i in range(5):
            # duplicate of image 0's content, new filename
            _write_file(source_dir / category / f"{category}_dup_{i}.jpg",
                        f"{category}-content-0".encode())

    split_dataset(str(source_dir), str(output_dir), seed=42)

    def hashes_in(split):
        hashes = set()
        for category in ["yes", "no"]:
            split_dir = output_dir / split / category
            for fname in os.listdir(split_dir):
                with open(split_dir / fname, 'rb') as f:
                    hashes.add(hashlib.md5(f.read()).hexdigest())
        return hashes

    train_hashes = hashes_in("train")
    val_hashes = hashes_in("val")
    test_hashes = hashes_in("test")

    assert train_hashes.isdisjoint(val_hashes)
    assert train_hashes.isdisjoint(test_hashes)
    assert val_hashes.isdisjoint(test_hashes)

    # duplicates should have been dropped before splitting: 20 unique
    # images per category, not 25
    assert len(train_hashes) + len(val_hashes) + len(test_hashes) == 40


def test_split_dataset_is_reproducible_with_same_seed(tmp_path):
    source_dir = tmp_path / "source"
    for category in ["yes", "no"]:
        os.makedirs(source_dir / category, exist_ok=True)
        for i in range(10):
            _write_file(source_dir / category / f"{category}_{i}.jpg",
                        f"{category}-content-{i}".encode())

    output_dir_1 = tmp_path / "output1"
    output_dir_2 = tmp_path / "output2"
    split_dataset(str(source_dir), str(output_dir_1), seed=7)
    split_dataset(str(source_dir), str(output_dir_2), seed=7)

    for split in ["train", "val", "test"]:
        for category in ["yes", "no"]:
            files_1 = sorted(os.listdir(output_dir_1 / split / category))
            files_2 = sorted(os.listdir(output_dir_2 / split / category))
            assert files_1 == files_2
