from config import BASE_DIR, DATA_DIR, LOG_DIR, ensure_runtime_dirs


def test_runtime_dirs_exist():
    ensure_runtime_dirs()
    assert BASE_DIR.exists()
    assert DATA_DIR.exists()
    assert LOG_DIR.exists()
