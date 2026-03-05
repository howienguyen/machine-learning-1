Below is a practical, end-to-end “known-good” guide for **WSL Ubuntu 24.04 + Intel Arc 140V (Lunar Lake) + Python 3.11 + TensorFlow 2.15.1 + ITEX 2.15.0.3**. It’s written to be reproducible: you can copy/paste blocks and know what each one is checking.

This follows Intel’s model for WSL GPU workflows: **Windows provides the GPU driver**, while **Linux-in-WSL provides the runtime components (Level Zero/OpenCL)**. ([Intel][1])

---

# 0) What “working” looks like

When everything is correct, you’ll see:

* `/dev/dxg` exists (WSL GPU bridge)
* `sycl-ls` shows a `[level_zero:gpu]` device
* TensorFlow shows `XPU:0` via ITEX

That’s your end state.

---

# 1) Windows host prerequisites (driver + WSL plumbing)

### 1.1 Update WSL (PowerShell, Admin)

```powershell
wsl --update
wsl --shutdown
```

Intel’s WSL GPU workflow guide assumes you’re on a current WSL2 build. ([Intel][1])

### 1.2 Install the latest Intel Graphics Windows driver

For WSL2 GPU compute, **the Windows Intel GPU driver is required** (WSL uses it under the hood). Intel’s Arc-on-WSL notes state drivers are installed on Windows, runtime in Linux. ([Intel GitHub][2])

After installing/updating the driver: **reboot Windows**.

---

# 2) WSL Ubuntu 24.04 prerequisites (GPU runtime)

## 2.1 Confirm the GPU bridge is present

Inside WSL:

```bash
ls -l /dev/dxg
uname -r
```

If `/dev/dxg` exists, Windows is exposing a GPU path into WSL. ([Intel][1])

## 2.2 Install Intel GPU runtime components (Level Zero + OpenCL)

On Ubuntu 24.04, Intel’s client GPU documentation points people to their packaging route (PPA/repo) for compute components (OpenCL, Level Zero). ([Intel DGPU Documentation][3])

### Option A (simple, often already enough): Ubuntu/Intel packages already present

Since your `clinfo` and `sycl-ls` already work, you may already have the runtime installed. Still, the canonical “core” packages are:

```bash
sudo apt update
sudo apt install -y \
  ocl-icd-libopencl1 clinfo \
  intel-opencl-icd \
  libze1
```

### Option B (if you hit dependency conflicts like libigc1 vs libigc2)

If you see conflicts similar to what you posted, it usually means you have a mixed “old IGC” vs “new IGC” set. Intel’s compute runtime packaging has multiple tracks and Ubuntu 24.04 can get into a conflict state. ([GitHub][4])

In that case, the clean strategy is: **pick one source of Intel compute packages and stick to it** (don’t mix random PPAs + random repos).

A commonly referenced approach for 24.04 is Intel’s `repositories.intel.com/gpu` key + repo setup (you’ll see this repeated in community guidance), but apply it only if you actually need it. ([Ask Ubuntu][5])

---

# 3) oneAPI tools (optional but very useful)

You don’t strictly need full oneAPI for TensorFlow XPU, but it’s great for diagnostics (`sycl-ls`, compilers, etc.).

If you installed `intel-oneapi-base-toolkit`, you should enable the environment each terminal:

```bash
source /opt/intel/oneapi/setvars.sh
```

Intel’s WSL GPU workflow page uses this “source setvars” pattern for oneAPI tools. ([Intel][1])

To make it automatic:

```bash
echo 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1' >> ~/.bashrc
```

---

# 4) Verify Level Zero / OpenCL from WSL

### 4.1 OpenCL device check

```bash
clinfo | grep -E "Platform Name|Device Name|Device Type" -n | head -n 140
```

You want to see an Intel graphics platform and a **GPU** device.

### 4.2 SYCL/Level Zero device check

```bash
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

You want a line like:
`[level_zero:gpu] ... Intel(R) Graphics ...`

This confirms the Level Zero backend is visible in WSL. ([Intel][1])

---

# 5) Python environment (pin the known-good combo)

Create a dedicated venv so system Python stays clean:

```bash
python3.11 -m venv .venv_itex
source .venv_itex/bin/activate
python -m pip install -U pip wheel setuptools
```

Install your known-good versions (pin them):

```bash
pip install tensorflow==2.15.1
pip install intel-extension-for-tensorflow==2.15.0.3
```

ITEX is the TensorFlow PluggableDevice plugin that exposes Intel XPU devices to TensorFlow. ([PyPI][6])

> Note: Intel documents the “XPU package” install as `pip install --upgrade intel-extension-for-tensorflow[xpu]`, but since you already know the exact working versions, pinning is the safer, reproducible route. ([Intel GitHub][7])

---

# 6) TensorFlow XPU validation script (must pass)

Run this:

```bash
source /opt/intel/oneapi/setvars.sh

python - <<'PY'
import tensorflow as tf
import intel_extension_for_tensorflow as itex
print("TF:", tf.__version__)
print("ITEX:", itex.__version__)
print("XPU devices:", tf.config.list_physical_devices("XPU"))
PY
```

Expected: `XPU devices: [PhysicalDevice(name='/physical_device:XPU:0', ...)]`

(Your earlier run already showed this, including ITEX GPU backend loaded.)

---

# 7) XPU compute smoke test (forces /XPU:0)

```bash
python - <<'PY'
import tensorflow as tf
import intel_extension_for_tensorflow as itex

print("XPU:", tf.config.list_physical_devices("XPU"))

with tf.device("/XPU:0"):
    a = tf.random.normal([2048, 2048])
    b = tf.random.normal([2048, 2048])
    c = tf.matmul(a, b)
    tf.print("matmul ok, shape:", tf.shape(c), "device:", c.device)
PY
```

If that prints a device containing XPU/SYCL, you’re genuinely executing on the Intel GPU.

---

# 8) Memory behavior controls (growth + cap)

### 8.1 Enable memory growth (if supported)

Put this at the start of your program **before any tensors/models**:

```python
xpus = tf.config.list_physical_devices("XPU")
for d in xpus:
    try:
        tf.config.experimental.set_memory_growth(d, True)
        print("Enabled memory growth on:", d)
    except Exception as e:
        print("Memory growth not supported on:", d, "->", e)
```

### 8.2 Cap XPU memory (if supported)

Also must be done **before runtime init**:

```python
XPU_MEMORY_LIMIT_MB = 4096
xpus = tf.config.list_physical_devices("XPU")
if xpus:
    try:
        tf.config.set_logical_device_configuration(
            xpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=XPU_MEMORY_LIMIT_MB)]
        )
        print("Logical XPUs:", tf.config.list_logical_devices("XPU"))
    except Exception as e:
        print("Could not cap XPU memory:", e)
```

Some backends honor these fully; others partially. The code is still the right first attempt.

---

# 9) Troubleshooting cheatsheet

### “XPU devices: []”

* Confirm `/dev/dxg` exists.
* `sycl-ls` must show `[level_zero:gpu]`.
* Ensure you’re importing ITEX: `import intel_extension_for_tensorflow as itex`.
* Confirm versions: `pip show tensorflow intel-extension-for-tensorflow`.

### APT dependency conflicts (libigc1 vs libigc2)

This is almost always a **mixed repo** situation. Choose one packaging source and purge conflicting packages so APT can resolve cleanly. Intel’s runtime packaging and Ubuntu 24.04 flows are particularly sensitive here. ([GitHub][4])

### Lots of CUDA warnings

That’s TensorFlow probing CUDA. It doesn’t block XPU.

---

# 10) Repro “lock file” suggestion

Create a `requirements.txt` for your working combo:

```txt
tensorflow==2.15.1
intel-extension-for-tensorflow==2.15.0.3
```

Then restore with:

```bash
pip install -r requirements.txt
```

---

[1]: https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2025-1/configure-wsl-2-for-gpu.html?utm_source=chatgpt.com "Configure WSL 2 for GPU Workflows"
[2]: https://intel.github.io/intel-extension-for-tensorflow/latest/docs/install/experimental/install_for_arc_gpu.html?utm_source=chatgpt.com "Intel® Arc™ A-Series GPU Software Installation"
[3]: https://dgpu-docs.intel.com/driver/client/overview.html?utm_source=chatgpt.com "Installing Client GPUs"
[4]: https://github.com/intel/compute-runtime?utm_source=chatgpt.com "Intel® Graphics Compute Runtime for oneAPI Level Zero ..."
[5]: https://askubuntu.com/questions/1543079/how-to-install-intel-arc-a380-drivers-on-ubuntu-24-04?utm_source=chatgpt.com "How to Install Intel ARC A380 Drivers on Ubuntu 24.04"
[6]: https://pypi.org/project/intel-extension-for-tensorflow/?utm_source=chatgpt.com "intel-extension-for-tensorflow"
[7]: https://intel.github.io/intel-extension-for-tensorflow/v2.15.0.1/docs/install/install_for_xpu.html?utm_source=chatgpt.com "Intel XPU Software Installation"
