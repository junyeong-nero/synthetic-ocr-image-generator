"""Capture fitting input validation and profile-compatible output."""
import argparse
import json

import pytest
import yaml

from src.cli.distribution import configure_parser
from src.cli.capture_fit import collect_pairs, aggregate_fits


def test_csv_paths_relative_to_manifest(tmp_path):
    manifest = tmp_path / "pairs.csv"
    manifest.write_text('clean,captured\na.png,b.jpg\n')
    assert collect_pairs(pairs=manifest) == [(tmp_path / 'a.png', tmp_path / 'b.jpg')]


def test_directory_matching_requires_same_names(tmp_path):
    clean, captured = tmp_path / 'clean', tmp_path / 'captured'
    clean.mkdir()
    captured.mkdir()
    for name in ['b.png', 'a.jpg']:
        (clean / name).touch()
        (captured / name).touch()
    assert [a.name for a, b in collect_pairs(clean_dir=clean, captured_dir=captured)] == ['a.jpg', 'b.png']
    (captured / 'b.png').unlink()
    with pytest.raises(ValueError, match='match'):
        collect_pairs(clean_dir=clean, captured_dir=captured)


@pytest.mark.parametrize('kwargs', [{}, {'clean_dir': 'x'}, {'pairs': 'x', 'captured_dir': 'y'}])
def test_invalid_sources(kwargs):
    with pytest.raises(ValueError):
        collect_pairs(**kwargs)


def test_aggregation_omits_unknown_values():
    result = aggregate_fits([{'params': {'dpi': 150, 'noise_sigma': None, 'grayscale': True}},
                             {'params': {'dpi': 200, 'noise_sigma': 3, 'grayscale': False}}], 'scanned')
    channel = result['capture_channels']['scanned']
    assert channel['dpi'] == {'choices': [150, 200]}
    assert channel['degradations']['noise_sigma'] == {'choices': [3]}
    assert channel['degradations']['grayscale'] == {'choices': [True, False]}


def test_cli_writes_results_and_reports_failed_pairs(tmp_path, monkeypatch):
    from src.realism import capture_fit
    manifest = tmp_path / 'pairs.csv'
    manifest.write_text('clean,captured\na.png,b.png\nc.png,d.png\n')
    def fake_fit(clean, captured, **kwargs):
        if clean.name == 'c.png':
            raise ValueError('cannot align')
        return {'params': {'dpi': 150, 'grayscale': True}}
    monkeypatch.setattr(capture_fit, 'fit_capture_files', fake_fit)
    output = tmp_path / 'out'
    parser = configure_parser(argparse.ArgumentParser())
    args = parser.parse_args(['fit-capture', '--pairs', str(manifest), '--output', str(output)])
    assert args.handler(args) == 1
    assert json.loads((output / 'pair-0001.json').read_text())['params']['dpi'] == 150
    assert 'cannot align' in (output / 'pair-0002.json').read_text()
    assert yaml.safe_load((output / 'profile.yaml').read_text())['capture_channels']['scanned']['dpi']


def test_real_cli_fit_outputs_sampleable_specs(tmp_path):
    import random
    from PIL import Image, ImageDraw
    from src.generator.distribution_profile import sample_value

    clean, captured = tmp_path / 'clean', tmp_path / 'captured'
    clean.mkdir()
    captured.mkdir()
    image = Image.new('RGB', (480, 640), 'white')
    draw = ImageDraw.Draw(image)
    rng = random.Random(7)
    for _ in range(100):
        x, y = rng.randrange(40, 400), rng.randrange(40, 550)
        draw.rectangle((x, y, x + rng.randrange(3, 30), y + 8), fill='black')
    image.save(clean / 'page.png')
    image.save(captured / 'page.png')
    output = tmp_path / 'output'
    args = configure_parser(argparse.ArgumentParser()).parse_args([
        'fit-capture', '--clean-dir', str(clean), '--captured-dir', str(captured),
        '--output', str(output), '--channel', 'photographed', '--clean-dpi', '200'])
    assert args.handler(args) == 0
    row = json.loads((output / 'pair-0001.json').read_text())
    assert row['params']['dpi'] == pytest.approx(200, abs=1)
    channel = yaml.safe_load((output / 'profile.yaml').read_text())['capture_channels']['photographed']
    assert sample_value(channel['dpi'], rng) == pytest.approx(200, abs=1)
    assert sample_value(channel['degradations']['binarize'], rng) is True


@pytest.mark.parametrize('contents', ['', 'clean,captured\n', 'foo,bar\na,b\n', 'clean,captured\na,\n'])
def test_malformed_or_empty_csv(tmp_path, contents):
    manifest = tmp_path / 'pairs.csv'
    manifest.write_text(contents)
    with pytest.raises(ValueError):
        collect_pairs(pairs=manifest)
