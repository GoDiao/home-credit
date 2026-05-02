"""
一键运行脚本：安装依赖 → 下载数据 → 跑 Pipeline → 启动 Dashboard

用法：
    python run.py          # 完整流程（首次运行）
    python run.py --skip   # 跳过数据下载和 Pipeline，直接启动 Dashboard
"""
import subprocess
import sys
import os
import time
import webbrowser
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
DASHBOARD = ROOT / "dashboard"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe" if sys.platform == "win32" else ROOT / ".venv" / "bin" / "python"


def banner(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def run(cmd, cwd=None, check=True):
    """运行命令，实时打印输出"""
    print(f"  > {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(
        cmd, cwd=cwd or ROOT, shell=isinstance(cmd, str),
        check=check, text=True,
    )
    return result


def check_python():
    """检查 Python 版本"""
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 11):
        print(f"[错误] 需要 Python >= 3.11，当前 {v.major}.{v.minor}.{v.micro}")
        sys.exit(1)
    print(f"  Python {v.major}.{v.minor}.{v.micro} OK")


def check_node():
    """检查 Node.js"""
    try:
        r = subprocess.run(["node", "--version"], capture_output=True, text=True, check=True)
        print(f"  Node.js {r.stdout.strip()} OK")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[错误] 未找到 Node.js，请先安装：https://nodejs.org/")
        sys.exit(1)


def check_uv():
    """检查 uv"""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        print("  uv OK")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[错误] 未找到 uv，请先安装：pip install uv")
        sys.exit(1)


def install_python_deps():
    """安装 Python 依赖"""
    banner("安装 Python 依赖")
    run(["uv", "sync"])


def install_frontend_deps():
    """安装前端依赖"""
    banner("安装前端依赖")
    if (DASHBOARD / "node_modules").exists():
        print("  node_modules 已存在，跳过")
        return
    run(["npm", "install"], cwd=DASHBOARD)


def check_data():
    """检查数据文件是否存在"""
    train_file = DATA_RAW / "application_train.csv"
    if train_file.exists():
        print("  数据已存在")
        return True

    print()
    print("  [错误] 未找到数据文件")
    print(f"  请先下载数据到 {DATA_RAW}")
    print()
    print("  下载地址：https://www.kaggle.com/c/home-credit-default-risk/data")
    print("  下载后解压 CSV 文件到 data/raw/ 目录")
    print()
    return False


def run_pipeline():
    """运行完整 Pipeline"""
    banner("运行 Pipeline（数据处理 → 特征工程 → 模型训练 → 策略 → 监控）")
    print("  这可能需要 10-30 分钟，取决于电脑性能...\n")
    run([sys.executable, "main.py", "--stage", "all"])


def start_backend():
    """启动后端"""
    print("  启动后端 (端口 8000)...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", "8000"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def start_frontend():
    """启动前端"""
    print("  启动前端 (端口 5174)...")
    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=DASHBOARD,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=True,
    )
    return proc


def wait_for_server(url, timeout=30):
    """等待服务就绪"""
    import urllib.request
    import urllib.error

    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description="Home Credit 风控建模项目 - 一键运行")
    parser.add_argument("--skip", action="store_true", help="跳过 Pipeline，直接启动 Dashboard")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    banner("Home Credit 风控建模项目 - 一键启动")
    print("  检查环境...")
    check_python()
    check_node()
    check_uv()

    if not args.skip:
        install_python_deps()
        install_frontend_deps()
        if not check_data():
            sys.exit(1)
        run_pipeline()
    else:
        # 即使 skip 也要确保依赖存在
        if not (ROOT / ".venv").exists():
            install_python_deps()
        if not (DASHBOARD / "node_modules").exists():
            install_frontend_deps()

    # 启动服务
    banner("启动 Dashboard")
    backend_proc = start_backend()

    # 等后端就绪
    print("  等待后端就绪...")
    if wait_for_server("http://localhost:8000/api/overview"):
        print("  后端就绪")
    else:
        print("  [警告] 后端启动超时，前端可能无法获取数据")

    frontend_proc = start_frontend()
    time.sleep(3)

    url = "http://localhost:5174"
    print(f"\n  Dashboard 地址: {url}")
    print("  按 Ctrl+C 退出\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  正在关闭...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("  已退出")


if __name__ == "__main__":
    main()
