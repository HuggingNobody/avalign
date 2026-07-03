import pytest

from avalign import cli


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "avalign" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_features_command(capsys):
    rc = cli.main(["features", "--samples", "4000", "--n-mels", "32", "--seed", "1"])
    assert rc == 0
    assert "对数梅尔谱形状" in capsys.readouterr().out


def test_retrieve_command(capsys):
    rc = cli.main(["retrieve", "--num-pairs", "40", "--dim", "32", "--seed", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "R@1" in out


def test_sync_command(capsys):
    rc = cli.main(["sync", "--length", "200", "--offset", "6", "--seed", "0"])
    assert rc == 0
    assert "估计偏移" in capsys.readouterr().out


def test_demo_command(capsys):
    rc = cli.main(["demo", "--num-pairs", "50", "--seed", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "端到端演示" in out
