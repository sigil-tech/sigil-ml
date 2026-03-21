class SigilMl < Formula
  include Language::Python::Virtualenv

  desc "ML prediction sidecar for the Sigil developer intelligence daemon"
  homepage "https://github.com/sigil-tech/sigil-ml"
  url "https://github.com/sigil-tech/sigil-ml/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "PLACEHOLDER"
  license "Apache-2.0"
  head "https://github.com/sigil-tech/sigil-ml.git", branch: "main"

  depends_on "python@3.12"

  resource "fastapi" do
    url "https://files.pythonhosted.org/packages/fastapi/fastapi-0.115.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "uvicorn" do
    url "https://files.pythonhosted.org/packages/uvicorn/uvicorn-0.32.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "scikit-learn" do
    url "https://files.pythonhosted.org/packages/scikit-learn/scikit-learn-1.5.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "joblib" do
    url "https://files.pythonhosted.org/packages/joblib/joblib-1.4.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/numpy/numpy-1.26.4.tar.gz"
    sha256 "PLACEHOLDER"
  end

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      sigil-ml is the ML prediction sidecar for sigild.

      To start manually:
        sigil-ml serve

      To configure sigild to use it, add to ~/.config/sigil/config.toml:
        [ml]
        mode = "local"
        [ml.local]
        enabled = true
        server_bin = "sigil-ml"

      Or re-run: sigild init
    EOS
  end

  service do
    run [opt_bin/"sigil-ml", "serve"]
    keep_alive true
    working_dir var
    log_path var/"log/sigil-ml.log"
    error_log_path var/"log/sigil-ml.log"
  end

  test do
    port = free_port
    pid = fork do
      exec bin/"sigil-ml", "serve", "--port", port.to_s
    end
    sleep 2
    output = shell_output("curl -s http://127.0.0.1:#{port}/health")
    assert_match "ok", output
  ensure
    Process.kill("TERM", pid) if pid
  end
end
